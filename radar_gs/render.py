import torch
import torch.nn.functional as F
from diff_gaussian_rasterization_radar import GaussianRasterizationSettings, GaussianRasterizer
from radar_gs.gaussian_model import GaussianModel


def render(viewpoint_camera, pc : GaussianModel, d_pc = None, energy = False, scaling_modifier = 1.0):
    """
    Render the specific cross-section of the 3D radar. 
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    if energy:
        # For energy calculation, we render the current spacial distribution of Gaussians in a single channel
        raster_settings = GaussianRasterizationSettings(
            image_channel=1,   
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.to(pc.device),
            viewdepth=viewpoint_camera.view_depth,
            prefiltered=False,
            debug=False
        )
    else:
        raster_settings = GaussianRasterizationSettings(
            image_channel=(viewpoint_camera.image_channel),   
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.to(pc.device),
            viewdepth=viewpoint_camera.view_depth,
            prefiltered=False,
            debug=False
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Get the means, intensity, scales and rotations of the Gaussians.
    if d_pc is None:
        means3D = pc.get_xyz
        intensity = pc.get_intensity
        means2D = screenspace_points
        scales = F.softplus(pc.get_scaling)
        rotations = pc.get_rotation
    else:
        # Add the deformations
        means3D = pc.get_xyz.detach() + d_pc.get_dxyz
        intensity = pc.get_intensity.detach() + d_pc.get_dintensity
        means2D = screenspace_points
        scales = F.softplus(pc.get_scaling.detach() + d_pc.get_dscaling)
        rotations = F.normalize(pc.get_rotation.detach() + d_pc.get_drotation)

    if energy:
        # Assume each Gaussian is of the same shape and intensity
        intensity = torch.ones((intensity.shape[0], 1), device=means3D.device, dtype=means3D.dtype) * 0.2
        scales = torch.ones_like(scales)
        rotations = torch.zeros_like(rotations)
        rotations[..., 0] += 1

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_radar, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        intensity = intensity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_radar,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
