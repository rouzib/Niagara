"""One double-source-plane image with independently evaluated light branches."""

from __future__ import annotations

from math import cos, sin, sqrt

import caustics as cs
import torch
from caskade import Module, forward

import niagara as ng


class FixedRedshiftMultiplane(cs.Multiplane):
    """Specialize the order of two fixed physical lens redshifts.

    Notes
    -----
    Only the redshift ordering is specialized. Deflections, lens-lens coupling
    and distance factors use the published Caustics implementation. Redshifts
    are fixed configuration and must not change after construction.
    """

    def get_z_ls(self):
        """Return the stored static redshifts without GPU scalar comparisons.

        Returns
        -------
        tuple of float
            Physical redshifts, unchanged from the stored Caustics parameters.
        """
        return self.fixed_redshifts


class MultiplaneImage(Module):
    """Render one image containing light from two different source planes.

    Parameters
    ----------
    size : int, optional
        Width and height of the single output image. Field of view is 4 arcsec.
    components : int, optional
        Number of Sersic components describing each physical source galaxy.
        Components are summed with LightStack; there is no batch dimension.
    quad_level : int, optional
        Gaussian integration order per pixel axis.
    device : str or torch.device, optional
        Home for this entire simulator instance.
    dtype : torch.dtype, optional
        Common model, parameter and coordinate dtype.
    fixed_order : bool, optional
        Specialize known redshift ordering for capture. False uses stock
        Multiplane sorting and provides an eager validation reference.

    Notes
    -----
    The mass planes lie at redshifts 0.4 and 0.9. The second plane contains a
    luminous galaxy; a background galaxy lies at redshift 2. Both galaxies
    contribute to the same observed field. Their integrated light is added
    into one (size, size) tensor. No PSF is applied in this intrinsic-image
    example. Independent workers must own separate instances of this class.

    The first SIE Einstein radius is defined for source redshift 2 in both
    branches. Its reduced deflection is rescaled using angular-distance ratios
    when tracing to the intermediate galaxy, preserving the same physical mass.
    """

    def __init__(
        self,
        size=256,
        components=8,
        quad_level=3,
        *,
        device="cpu",
        dtype=torch.float32,
        fixed_order=True,
    ):
        if min(size, components, quad_level) < 1:
            raise ValueError("Image size, component count and quadrature order must be positive")
        super().__init__("scene")
        self.cosmology = cs.FlatLambdaCDM(name="cosmology")
        self.foreground = cs.SIE(
            self.cosmology,
            name="foreground",
            z_l=0.4,
            x0=-0.12,
            y0=0.04,
            q=0.78,
            phi=0.3,
            Rein=1.05,
        )
        self.secondary = cs.SIE(
            self.cosmology,
            name="secondary",
            z_l=0.9,
            x0=0.08,
            y0=0.03,
            q=0.84,
            phi=-0.2,
            Rein=0.25,
        )
        for lens in (self.foreground, self.secondary):
            for name in ("x0", "y0", "q", "phi", "Rein"):
                lens.children[name].to_dynamic()
        kind = FixedRedshiftMultiplane if fixed_order else cs.Multiplane
        self.mass = kind(self.cosmology, (self.foreground, self.secondary), z_s=2.0, name="mass")
        if fixed_order:
            self.mass.fixed_redshifts = tuple(float(lens.z_l.value) for lens in self.mass.lenses)
        self.middle_light = self._light(components, "middle", 0.08, 0.03, 0.3)
        self.background_light = self._light(components, "background", 0.02, 0.07, 0.15)
        pixelscale = 4.0 / size
        x, y = cs.utils.meshgrid(pixelscale, size, dtype=dtype)
        self._x, self._y, self._weights = cs.utils.gaussian_quadrature_grid(
            pixelscale, x, y, quad_level=quad_level
        )
        self.to(device=device, dtype=dtype)
        # SIE deflections are normalized to the final source at z=2.0.
        z_l = self.foreground.z_l.value
        z_m = self.secondary.z_l.value
        z_s = self.mass.z_s.value
        self._middle_ratio = (
            self.cosmology.angular_diameter_distance_z1z2(z_l, z_m)
            / self.cosmology.angular_diameter_distance(z_m)
            * self.cosmology.angular_diameter_distance(z_s)
            / self.cosmology.angular_diameter_distance_z1z2(z_l, z_s)
        )

    @staticmethod
    def _light(count, prefix, x0, y0, radius):
        lights = []
        for i in range(count):
            angle = i * 2.399963229728653
            offset = radius * sqrt(i / max(count - 1, 1))
            light = cs.Sersic(
                name=f"{prefix}_{i}",
                x0=x0 + offset * cos(angle),
                y0=y0 + offset * sin(angle),
                q=0.65 + 0.1 * cos(angle),
                phi=angle,
                n=1.0 + 0.3 * (i % 3),
                Re=radius if i == 0 else radius / 4,
                Ie=1.0 if i == 0 else 0.25,
            )
            for name in ("x0", "y0", "q", "phi", "n", "Re", "Ie"):
                light.children[name].to_dynamic()
            lights.append(light)
        return cs.LightStack(lights, name=f"{prefix}_light")

    def to(self, device=None, dtype=None):
        """Move simulator state and fixed integration coordinates together.

        Parameters
        ----------
        device : str or torch.device, optional
            Destination for model parameters and integration coordinates.
        dtype : torch.dtype, optional
            Common floating-point dtype.

        Returns
        -------
        MultiplaneImage
            This simulator instance.
        """
        super().to(device=device, dtype=dtype)
        for name in ("_x", "_y", "_weights", "_middle_ratio"):
            if hasattr(self, name):
                setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))
        return self

    @forward
    def middle(self):
        """Render the intermediate galaxy lensed by the foreground mass.

        Returns
        -------
        torch.Tensor
            Integrated intermediate-plane light on the observed pixel grid.
        """
        ax, ay = self.foreground.reduced_deflection_angle(self._x, self._y)
        brightness = self.middle_light.brightness(
            self._x - self._middle_ratio * ax, self._y - self._middle_ratio * ay
        )
        return cs.utils.gaussian_quadrature_integrator(brightness, self._weights)

    @forward
    def background(self):
        """Render the background galaxy with full two-plane ray tracing.

        Returns
        -------
        torch.Tensor
            Integrated background light after coupled two-plane ray tracing.
        """
        bx, by = self.mass.raytrace(self._x, self._y)
        brightness = self.background_light.brightness(bx, by)
        return cs.utils.gaussian_quadrature_integrator(brightness, self._weights)

    @forward
    def __call__(self):
        """Return one image containing both physically distinct light planes.

        Returns
        -------
        torch.Tensor
            The single observed image, with shape ``(size, size)``.
        """
        return self.middle() + self.background()


class ParallelMultiplaneImage:
    """Evaluate disjoint light branches and return their single summed image.

    Parameters
    ----------
    middle, background : MultiplaneImage
        Separately owned copies of the same physical scene, already on their
        device homes. The middle copy supplies intermediate light; the other
        copy supplies background light. Both accept the same packed parameters.
    example : torch.Tensor
        Packed scalar parameters, without a batch dimension.
    compiled : bool, optional
        Compile each pure branch with Niagara before worker submission.

    Notes
    -----
    The output resides on the background instance's home. Transferring the
    intermediate contribution and adding it are part of every forward call.
    Preparation finishes before workers execute. This class owns its executor;
    use it as a context manager. Models must remain fixed during use.
    """

    def __init__(self, middle, background, example, *, compiled=False):
        if middle is background:
            raise ValueError("Independent workers must own different simulator instances")
        if example.ndim != 1:
            raise ValueError("Pass one packed parameter vector without a batch dimension")
        self.models = (middle, background)
        self.homes = (middle._x.device, background._x.device)
        self.functions = (middle.middle, background.background)
        if compiled:
            self.functions = tuple(
                ng.compile_model(model, example.to(home), method=method)
                for model, home, method in zip(self.models, self.homes, ("middle", "background"))
            )
        self.executor = ng.AsyncGraphExecutor()

    def __call__(self, params):
        """Return the combined image after ordering both producer streams.

        Parameters
        ----------
        params : torch.Tensor
            One packed physical parameter vector, without a batch dimension.

        Returns
        -------
        torch.Tensor
            One combined image on the background worker's device.
        """
        if params.ndim != 1:
            raise ValueError("Pass one packed parameter vector without a batch dimension")
        tasks = tuple(
            self.executor.submit(function, params, device=home)
            for function, home in zip(self.functions, self.homes)
        )
        middle, background = self.executor.resolve(tasks)
        return middle.to(self.homes[1], non_blocking=True) + background

    def close(self):
        """Finish submitted jobs and release the persistent workers."""
        self.executor.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
