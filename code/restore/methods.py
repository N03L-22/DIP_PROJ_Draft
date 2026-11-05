"""STAGE 3 registry - every restoration method behind one interface:  method(image) -> restored image.

The experiment runner reads method names and settings from configs/<medium>.json and asks this file for the matching
function. Adding a method means writing it once and adding one line to REGISTRY.
"""
from restore import baselines, extra, frequency, fusion, medium_aware
from restore.dcp import dcp_restore

REGISTRY = {
    # name: function taking (image, **settings) and returning a restored picture
    "none":               lambda image, **s: image,                                             # no restoration: the score to beat
    "dcp":                lambda image, **s: dcp_restore(image, refine=False, **s)["restored"],  # plain DCP, blocky transmission
    "dcp_guided":         lambda image, **s: dcp_restore(image, refine=True, **s)["restored"],   # DCP + guided filter
    "wb_dcp":             medium_aware.wb_then_dcp,                                              # white balance, then DCP
    "udcp":               medium_aware.udcp,                                                     # underwater: ignore red
    "dcp_local_airlight": medium_aware.dcp_local_airlight,                                       # glass: uneven film
    "dcp_near_global":    medium_aware.dcp_near_global,                                          # glass: almost constant t
    "homomorphic":        baselines.homomorphic,                                                 # glass: log, filter, exp
    "wiener":             baselines.wiener_deblur,                                               # plastic: undo the blur
    "retinex":            baselines.retinex_msr,                                                 # multi-scale Retinex
    "clahe":              baselines.clahe,                                                       # local contrast
    "hist_eq":            baselines.histogram_equalise,                                          # global histogram equalisation (lightness)
    "hist_eq_rgb":        baselines.histogram_equalise_rgb,                                      # ... on each colour separately
    "gamma_auto":         baselines.auto_gamma,                                                  # power curve to a target brightness
    "stretch":            baselines.contrast_stretch,                                            # linear contrast stretch
    "stretch_rgb":        baselines.contrast_stretch_per_channel,                                # stretch with a separate range per colour (inverse of a per-colour affine map)
    "dcp_clahe":          medium_aware.dcp_then_clahe,                                           # DCP then CLAHE
    # --- colour and tone (extra.py)
    "gray_world":         lambda image, **s: baselines.gray_world(image),                        # white balance: gray-world
    "white_patch":        extra.white_patch,                                                     # white balance: brightest = white
    "shades_of_gray":     extra.shades_of_gray,                                                  # white balance: p-norm mean
    "von_kries":          lambda image, **s: extra.von_kries(image),                             # white balance in cone space
    "red_comp":           extra.red_channel_compensation,                                        # underwater: restore red
    "eq_hsi":             lambda image, **s: extra.equalise_intensity_hsi(image),                # equalise intensity only
    "eq_avg_hist":        lambda image, **s: extra.equalise_to_average_histogram(image),         # equalise to the average histogram
    "hist_spec":          extra.histogram_specification,                                         # target a bell-shaped histogram
    "log":                lambda image, **s: extra.log_transform(image),                         # log transform
    "power_law":          extra.power_law,                                                       # fixed gamma
    # --- local contrast
    "local_stats":        extra.local_statistics_enhance,                                        # enhance dark, dull regions only
    "flat_field":         extra.flat_field,                                                      # divide out a slow film
    "top_hat":            extra.top_hat_background,                                              # morphological background removal
    # --- sharpening, deblurring, cleanup
    "unsharp":            extra.unsharp_mask,                                                    # unsharp masking / high-boost
    "freq_laplacian":     extra.frequency_laplacian_sharpen,                                     # sharpen in the frequency domain
    "richardson_lucy":    extra.richardson_lucy,                                                 # iterative deblur
    "cls":                extra.constrained_least_squares,                                       # constrained least squares deblur
    "median":             extra.median_cleanup,                                                  # median filter
    "adaptive_nr":        extra.adaptive_noise_reduction,                                        # adaptive local noise reduction
    # --- multi-scale fusion and Tarel
    "fusion":             fusion.fusion_enhance,                                                 # Laplacian-pyramid fusion
    "fusion_dcp":         fusion.fusion_after_dcp,                                               # fuse original, DCP, CLAHE, unsharp
    "tarel":              fusion.tarel_veil_removal,                                             # median-of-medians veil removal
    # --- haze removal by subtraction and frequency-domain filtering (frequency.py)
    "lowfreq_subtract":   frequency.lowfreq_subtract,                                            # subtract the slow veil, stretch
    "blind_veil":         frequency.blind_veil_inversion,                                        # J = (I - V) / t with t, V from priors
    "blind_veil_local":   lambda image, **s: frequency.blind_veil_inversion(image, local_sigma=0.1),  # the same with a veil that follows the picture
}


def build_method(spec):
    """Turn {"name": "dcp", "params": {...}} from a config file into a function of one argument (the picture)."""
    function = REGISTRY[spec["name"]]
    settings = spec.get("params", {})
    return lambda image: function(image, **settings)


def build_sequence(steps):
    """Chain several methods: each step's output is the next step's input. `steps` is a list of names or of specs like {"name": ..., "params": {...}}."""
    methods = [build_method(step if isinstance(step, dict) else {"name": step}) for step in steps]

    def run(image):
        for method in methods:
            image = method(image)
        return image
    return run
