"""Optional, reproducible Caustics workloads. Importing Niagara does not require Caustics."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import torch
from caskade import Module


@dataclass
class ImageProblem:
    """Hold the classic image simulator and its physical/latent parameter map.

    Parameters
    ----------
    model : caskade.Module
        Existing Caustics LensSource with packed dynamic parameters.
    truth : torch.Tensor
        Synthetic physical parameter vector in the actual Caskade packing order.
    names : tuple of str
        Labels matching packed physical scalar parameters.
    center, scale : torch.Tensor
        Prior coordinate centers/scales used before physical transforms.
    kinds : tuple of str
        Per-component transforms: ``"normal"``, ``"log"``, ``"q"`` or ``"n"``.

    Notes
    -----
    Standard-normal latent priors are mapped to physical parameters. They are
    configured before synthetic data and need not have their median at truth.
    Transforming density coordinates also requires the appropriate prior/target
    definition; these helpers only encode/decode parameter values.
    """

    model: Module
    truth: torch.Tensor
    names: tuple[str, ...]
    center: torch.Tensor
    scale: torch.Tensor
    kinds: tuple[str, ...]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent coordinates to physical packed parameters.

        Parameters
        ----------
        z : torch.Tensor
            Latent values with the parameter dimension last; leading batch
            dimensions are preserved.

        Returns
        -------
        torch.Tensor
            Physical values in model packing order. Positive quantities use an
            exponential, q uses a sigmoid, and n lies between 0.36 and 10.
        """
        values = []
        for i, kind in enumerate(self.kinds):
            x = self.center[i] + self.scale[i] * z[..., i]
            if kind == "log":
                x = x.exp()
            elif kind == "q":
                x = x.sigmoid()
            elif kind == "n":
                x = 0.36 + 9.64 * x.sigmoid()
            values.append(x)
        return torch.stack(values, dim=-1)

    def encode(self, theta: torch.Tensor) -> torch.Tensor:
        """Invert the physical parameter map on its valid open domain.

        Parameters
        ----------
        theta : torch.Tensor
            Physical values with parameter dimension last. Positive quantities,
            q in (0, 1), and n in (0.36, 10) are required for finite inverse maps.

        Returns
        -------
        torch.Tensor
            Latent coordinates with matching leading dimensions.
        """
        values = []
        for i, kind in enumerate(self.kinds):
            x = theta[..., i]
            if kind == "log":
                x = x.log()
            elif kind == "q":
                x = torch.logit(x)
            elif kind == "n":
                x = torch.logit((x - 0.36) / 9.64)
            values.append((x - self.center[i]) / self.scale[i])
        return torch.stack(values, dim=-1)


def image_problem(
    size: int = 60,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    psf_mode: str = "conv2d",
) -> ImageProblem:
    """Build the classic 21-parameter Caustics lens-image workload.

    Parameters
    ----------
    size : int, optional
        Image width/height in pixels. Pixel scale remains 0.05 arcsec, so larger
        arrays cover a wider field rather than refine a fixed field of view.
    device : str or torch.device, optional
        Initial common device for model state and returned parameter tensors.
    dtype : torch.dtype, optional
        Computation/parameter dtype, defaulting to float64.
    psf_mode : {"conv2d", "fft"}, optional
        Native Caustics PSF convolution mode.

    Returns
    -------
    ImageProblem
        SIE mass plus external shear, Sersic source and lens light, an integrated
        normalized 7 x 7 Gaussian PSF, truth vector and prior-coordinate metadata.

    Notes
    -----
    Requires the optional Caustics dependencies. Parameter order is derived from
    actual Param identities instead of constructor keyword order. This creates
    a workload and prior map, not observations or accepted posterior samples.
    """
    import caustics as cs

    cosmo = cs.FlatLambdaCDM(name="cosmo")
    mass = cs.SIE(cosmo, name="mass", x0=0.05, y0=0.0, q=0.86, phi=-0.2, Rein=0.66)
    shear = cs.ExternalShear(cosmo, name="shear", x0=0.0, y0=0.0, gamma_1=0.0, gamma_2=-0.05)
    lens = cs.SinglePlane(cosmo, lenses=(mass, shear), z_l=0.5, z_s=1.5, name="lens")
    source = cs.Sersic(
        name="source", x0=0.1, y0=0.0, q=0.75, phi=1.18, n=1.0, Re=0.1 / sqrt(0.75), Ie=0.04
    )
    light = cs.Sersic(
        name="light", x0=0.05, y0=0.0, q=0.75, phi=1.18, n=2.0, Re=0.6 / sqrt(0.75), Ie=0.04
    )
    # Integrate the Gaussian on an upsampled grid, then bin to 7x7.
    fine = (torch.arange(14, dtype=dtype) - 6.5) * 0.05 / 2
    yy, xx = torch.meshgrid(fine, fine, indexing="ij")
    sigma = 0.05 / (2 * sqrt(2 * log(2)))
    psf = torch.nn.functional.avg_pool2d(
        torch.exp(-(xx.square() + yy.square()) / (2 * sigma**2))[None, None], 2
    )[0, 0]
    psf /= psf.sum()
    model = cs.LensSource(
        lens, source, 0.05, size, lens_light=light, psf=psf, psf_mode=psf_mode, name="sim"
    )
    specs = {}
    for module, keys in (
        (mass, ("x0", "y0", "q", "phi", "Rein")),
        (shear, ("gamma_1", "gamma_2")),
        (source, ("x0", "y0", "q", "phi", "n", "Re", "Ie")),
        (light, ("x0", "y0", "q", "phi", "n", "Re", "Ie")),
    ):
        for key in keys:
            param = module.children[key]
            truth = param.value.detach().clone()
            if key in ("x0", "y0"):
                kind, center, scale = "normal", 0.0, 0.2
            elif key.startswith("gamma"):
                kind, center, scale = "normal", 0.0, 0.1
            elif key == "phi":
                kind, center, scale = "normal", 0.0 if module is mass else 1.0, 0.7
            elif key == "q":
                kind, center, scale = "q", torch.logit(torch.tensor(0.8)).item(), 0.8
            elif key == "n":
                median = 1.0 if module is source else 2.0
                kind, center, scale = (
                    "n",
                    torch.logit(torch.tensor((median - 0.36) / 9.64)).item(),
                    0.6,
                )
            else:
                median, scale = {
                    "Rein": (0.7, 0.35),
                    "Re": (0.13 if module is source else 0.7, 0.4),
                    "Ie": (0.04, 0.7),
                }[key]
                kind, center = "log", log(median)
            specs[param] = (f"{module.name}.{key}", truth, kind, center, scale)
            param.to_dynamic()
    rows = [specs[param] for param in model.dynamic_params]
    model.to(device=device, dtype=dtype)
    return ImageProblem(
        model,
        torch.stack([r[1] for r in rows]).to(device=device, dtype=dtype),
        tuple(r[0] for r in rows),
        torch.tensor([r[3] for r in rows], device=device, dtype=dtype),
        torch.tensor([r[4] for r in rows], device=device, dtype=dtype),
        tuple(r[2] for r in rows),
    )


def pixelated_scene(
    size: int = 64,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[Module, torch.Tensor]:
    """Build a pixelated convergence field with a dynamic Sersic source.

    Parameters
    ----------
    size : int, optional
        LensSource image width/height in pixels at 0.05 arcsec per pixel.
    dtype : torch.dtype, optional
        Grid and model dtype.

    Returns
    -------
    model : caskade.Module
        Native Caustics LensSource, ready for an explicit device plan.
    params : torch.Tensor
        Seven Sersic parameters in the model's actual dynamic packing order.

    Notes
    -----
    The 32 x 32 convergence field is static. This fixture separates the field
    and source into regions without replacing Caustics' numerical operations.
    """
    import caustics as cs

    cosmo = cs.FlatLambdaCDM(name="cosmo")
    axis = torch.linspace(-1.5, 1.5, 32, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kappa = 0.12 * torch.exp(-(xx.square() + yy.square()) / 0.2)
    field = cs.PixelatedConvergence(
        0.1, cosmo, z_l=0.5, z_s=1.5, convergence_map=kappa, name="field"
    )
    source = cs.Sersic(x0=0.08, y0=0.03, q=0.8, phi=0.4, n=1.0, Re=0.2, Ie=1.0, name="source")
    model = cs.LensSource(field, source, 0.05, size, name="sim")
    source.to_dynamic()
    model.to(dtype=dtype)
    values = torch.tensor([0.08, 0.03, 0.8, 0.4, 1.0, 0.2, 1.0], dtype=dtype)
    # Return values in Caskade's order, which need not match constructor order.
    by_name = dict(zip(("x0", "y0", "q", "phi", "n", "Re", "Ie"), values))
    params = torch.stack([by_name[p.name] for p in model.dynamic_params])
    return model, params


def multiplane_scene(
    size: int = 48,
    *,
    dtype: torch.dtype = torch.float64,
    fixed_order: bool = True,
) -> tuple[Module, torch.Tensor]:
    """Build the 22-scalar, two-plane lensing validation scene.

    Parameters
    ----------
    size : int, optional
        Width/height of the sampled image grid.
    dtype : torch.dtype, optional
        Model, grid and packed-parameter dtype.
    fixed_order : bool, optional
        Specialize the known static redshift order for capture. False retains
        stock Caustics Multiplane sorting as an eager reference.

    Returns
    -------
    model : caskade.Module
        Scene containing two physical lens planes, two sources, pixelated mass,
        EPL components, shear, multipoles and tied intermediate-source positions.
    params : torch.Tensor
        Twenty-two physical scalar values in Caskade's packed order.

    Notes
    -----
    The fixed-order variant preserves stored static redshifts and specializes
    only their already-known ordering. Validate it against ``fixed_order=False``
    before using this fixture as a reference. It is not an automatic solver for
    data-dependent redshift ordering.
    """
    import caustics as cs
    from caskade import Module, forward

    class FixedOrder(cs.Multiplane):
        def get_z_ls(self):
            return self.fixed_redshifts

    class Scene(Module):
        def __init__(self):
            super().__init__("scene")
            self.cosmo = cs.FlatLambdaCDM(name="cosmo")
            self.primary = cs.EPL(
                self.cosmo,
                x0=0.03,
                y0=-0.02,
                q=0.8,
                phi=0.2,
                Rein=0.7,
                t=1.05,
                name="primary",
                n_iter=8,
            )
            self.shear = cs.ExternalShear(
                self.cosmo, x0=0.0, y0=0.0, gamma_1=0.03, gamma_2=-0.02, name="shear"
            )
            self.multipoles = cs.Multipole(
                self.cosmo,
                m=(3, 4),
                x0=0.0,
                y0=0.0,
                a_m=torch.tensor([0.006, 0.004]),
                phi_m=torch.tensor([0.2, 0.4]),
                name="multipoles",
            )
            axis = torch.linspace(-1.5, 1.5, 16, dtype=dtype)
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            self.field = cs.PixelatedConvergence(
                0.2,
                self.cosmo,
                name="field",
                convergence_map=0.03 * torch.exp(-(xx.square() + yy.square()) / 0.4),
            )
            self.foreground = cs.SinglePlane(
                self.cosmo,
                (self.primary, self.field, self.shear, self.multipoles),
                z_l=0.4,
                name="foreground",
            )
            self.middle = cs.Sersic(
                x0=0.11, y0=0.04, q=0.75, phi=0.6, n=1.0, Re=0.13, Ie=0.3, name="middle"
            )
            self.perturber = cs.EPL(
                self.cosmo,
                z_l=0.9,
                x0=0.0,
                y0=0.0,
                q=0.9,
                phi=0.1,
                Rein=0.08,
                t=1.0,
                name="perturber",
                n_iter=8,
            )
            self.perturber.x0 = self.middle.x0
            self.perturber.y0 = self.middle.y0
            self.background = (FixedOrder if fixed_order else cs.Multiplane)(
                self.cosmo, (self.foreground, self.perturber), z_s=2.0, name="background"
            )
            self.source = cs.Sersic(
                x0=0.03, y0=0.01, q=0.8, phi=0.2, n=1.2, Re=0.1, Ie=0.5, name="source"
            )
            self.grid = cs.utils.meshgrid(0.05, size)
            free = [
                self.primary.Rein,
                self.primary.q,
                self.shear.gamma_1,
                self.shear.gamma_2,
                self.multipoles.a_m,
                self.field.scale,
                *self.middle.children.values(),
                self.perturber.Rein,
                *self.source.children.values(),
            ]
            for param in free:
                param.to_dynamic()

        def to(self, device=None, dtype=None):
            super().to(device=device, dtype=dtype)
            self.grid = tuple(t.to(device=device, dtype=dtype) for t in self.grid)
            return self

        @forward
        def __call__(self):
            bx, by = self.foreground.raytrace(*self.grid, z_s=self.grid[0].new_tensor(0.9))
            intermediate = self.middle.brightness(bx, by)
            bx, by = self.background.raytrace(*self.grid)
            return intermediate + self.source.brightness(bx, by)

    model = Scene().to(dtype=dtype)
    if fixed_order:
        # Preserve constructor rounding of the stored static redshifts exactly.
        model.background.fixed_redshifts = tuple(
            float(lens.z_l.value) for lens in model.background.lenses
        )
    return model, model.get_values().detach().clone()
