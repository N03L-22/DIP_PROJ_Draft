"""Forward degradation models: haze, water, glass, plastic, and a stack that chains them.

Every model takes a clean image J and returns the degraded image, using the same idea:
    I = J * t + A * (1 - t)
where t is the fraction of scene light that gets through and A is the light the medium adds.
Images are float arrays of shape (height, width, 3) with values from 0 to 1.
"""
