![Logo](http://sebastian.stapelberg.de/documents/quicklens.jpg "Logo")

# quicklens

Quicklens is a gravitational lensing visualization tool based on the [OpenCV](https://opencv.org/) image processing library.

(See [**DEMO VIDEO**](https://youtu.be/b1gHwyzFu8g))

## Getting started

These are the instructions in order to setup and run the code on your local machine:

### Requirements 
We will need the following libraries:
- OpenCV (>= v2)
- cfitsio, CCfits: \*.FITS image reading (optionally, you can disable it in the Makefile)
- OpenMP (libgomp): multiprocessing

The Makefile assumes that pkg-config is installed.

### Setup and usage

Run the Makefile (where the option "USE_CCFITS" can be modified to toggle compilation with the CCfits library). 

This should create a binary "lens". Call "lens" with:

```shell
$ ./lens  LENS  SOURCE [N_threads (default: all)]
```
where:
- LENS is an image containing the mass distribution of the lens, projected along the line of sight in dimensionless units (also called "convergence"). This can be either a \*.FITS-image or usual image formats like \*.PNG, \*.JPG etc. In the latter case, quicklens will use the grayscale intensity for setting the convergence. 
- SOURCE is the image of the source (to be lensed), given as RGB image (\*.PNG, \*.JPG, etc). 
- N_threads is an optional argument to set the number of threads used for the image rendering. The default is to use all.

The lens can be dragged around with the left mouse key. In addition, there are several trackbars to adjust the image or display physics-related information.


Please note:

- There is a **directory containing ready example images** for lenses and sources.
- It is advisable to have roughly the same sizes for the lens and source images. A good choice for the image dimensions are screen dimensions e.g. 1024x768 or 1024x576 for 16:9 screens. For a HD display and a recent multi-core CPU, I usually apply 1920x1080.
- The lensing effect is highly sensitive to the overall amount and distribution of surface density. Already a small blob with a ***tiny*** intensity can result in large deflections. An image with multiple blobs may require to change the normalization by the "kappa_weight" parameter; and images with large surfaces of constant density may lead to strange results as well.

Have fun!

## Documentation

The code itself is documented in "doc" (auto-generated using doxygen).

## Limitations

- Please pay attention to finite area effects at the boundaries. If the lens convergence does not fall off quickly towards the image boundaries (e.g. if there are multiple highly extended lenses or a smooth mass distribution over the entire image), the missing information can affect the accuracy in the calculation of the deflection.

- Note also that in this regard, special attention has to be paid to the definition of the deflection angle and the difference it makes whether a lensing halos is isolated or embedded in a multi-halo environment. Even for comparably large inter-halo-distances, the latter can produce an additional component in the deflection angle that increases quadratically on average (as one would expect for a constant density background, see e.g. Narayan & Bartelmann 1996).

- The program uses discrete Fourier methods to approximate the superposition. In theory, this could introduce tiny unwanted noise on top of the deflection or the critical curves. Please pay attention to this effect.

## Science FAQ

- **What is gravitational lensing?**
    - Light rays get deflected in gravitational fields. This causes images of distant objects to appear focused and distorted when viewn through a dense foreground matter structure in the line of sight. The foreground object thus acts similar to a lens. According to General Relativity, this is a consequence of curvature in the four-dimensional spacetime structure, caused by the presence of the lens matter.

- **What is strong gravitational lensing?**
    - When the density of the lens is sufficient, its gravity can produce multiple images of a single source, appearing on different positions on the sky. These images are often highly distorted, appearing as elongated arcs if the source object has a sufficient angular extent. In this case one speaks of "strong lensing". Its opposite, weak lensing, refers to the case in which the lens is either too weak or too far away from our line of sight to the background object to produce multiple images. In this case the lens produces only a single, slightly distorted image.

   
- **Does this effect occur often?**
    - Sure, we can see it all over the sky, in the form of small distortions in astronomical images. In the case of most stars and galaxies, the gravitational light deflection only causes small displacements of the object positions on the sky. However, if a massive galaxy or a galaxy cluster happens to be exactly in front of a background object from our viewing angle, we can see the formation of arcs (banana-shaped distortions) or even rings.

- **Does that mean we don't see stars and galaxies at their real positions on the sky?**
   - Yes, this is exactly what it means. But the effect is tiny. Even in the case of strong lensing, the displacement of apparent positions is of the order micro-arcseconds. Nonetheless, the effect is seen in many spots all over the sky and its magnitude is sufficient for astronomers to use the observed image distortions in order to infer how much mass is present in the universe, both in the form of dark and luminous matter. The study of lensing helps cosmologists to explore the properties of dark matter and dark energy, or to measure cosmological parameters such as the present expansion rate of the universe. In addition, strong lenses act as "natural telescopes" by magnifying the brightness of otherwise undetectable objects. This is used frequently to study the oldest objects in the universe.

- **What are the 'critical curves' displayed in the quicklens 'overlays'?**
   - Critical curves are those places in the sky in which the lens mapping becomes "singular", meaning that it leads to degenerate solutions separating multiple images if there are any. Thus, critical curves are the places near which the ring-like distortions caused by strong lenses form. The interior area enclosed by a critical curve in a certain sense contains a "mirror" version of the exterior images. The points on which a source pixel must lie in order for its lensed image to appear on a critical curve are called caustics. In *quicklens*, caustics and critical lines can be shown as overlays. The caustics are the red lines, while the critical curves are drawn as white lines. Notice that moving the lens such that a source image pixel crosses one of the red caustics causes this pixel to appear (or disappear) on additional locations.

- **How does the lensing visualization work?**
   - The quicklens code solves the ray-tracing equations for lensing, which describe how the angular positions of light rays received from distant locations in the sky changes along the light path between source and observer. This requires knowledge of the perpendicular acceleration vectors acting on each photon along its light path. 
   For all relevant applications of lensing, it is sufficient to treat the equations in the so-called limit of weak gravitational fields and Born's approximation for small deflection angles, in which the ten highly non-linear, coupled gravity field equations simplify drastically to a numerically solvable problem – and in which the deflection vectors at different z-positions are uncorrelated. This is why a 2D image of the line-of-sight-projected mass distribution passed by the user is sufficient, while the 3D information is irrelevant for lensing. We can then easily compute a "deflection field" quantifying the sum of changes in direction due to gravitational accelerations for every photon. When the user moves the lens over the screen, we simply use this deflection field to map source pixels to the image pixels on the screen. In this way, we can construct the distorted image as it looks viewn through the gravity field of the user-defined matter distribution.

- **Can you explain the math in more detail?**
   - For the physics experts: We compute the effective lensing potential "psi" from the user-defined convergence "kappa" using Fourier methods to do the superposition, i.e. sum all gravity contributions from the various points. Differentiating the lensing potential once yields the deflection angle field, which is used to solve the lens equation for the ray-tracing. The derivatives of the deflection field are furthermore used to compute the Jacobian matrix of this equation to study how the light rays are focused and sheared; and where the lens equation has its singularities. This information can be displayed on top of the image using the trackbars. For more information on lensing, see for instance the reviews by Bartelmann et al.


## Weblinks

http://sebastian.stapelberg.de

