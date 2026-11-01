# Procedural head geometry

Original geometry generated for this app from an ellipsoid equation with a
tapered lower jaw. No original the reference application coordinates, scans, textures,
learned weights, or third-party head assets were used.

Format: nine decimal coordinates per row, representing three triangle vertices.
Axes: X horizontal, Y vertical, Z depth. Arbitrary model units.
Resolution: 16 latitude subdivisions, 32 longitude subdivisions, 960 triangles.
Parameters: X radius 0.72, Y radius 1, Z radius 0.82;
X jaw taper = 1 - 0.18 * max(0, -Y).

This project dedicates this generated geometry to the public domain under
CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/
No attribution is requested for this generated asset.

The runtime currently uses the mesh bounds as a modest head-pose smoothing
reference, capped at 10% for this generic model. A registered personal profile
takes precedence and retains its 18% cap. This is not a fitted anatomical model
or a demonstrated improvement in gaze accuracy.
