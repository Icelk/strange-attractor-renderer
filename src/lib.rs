//! Master branch online documentation is available at
//! [doc.icelk.dev](https://doc.icelk.dev/strange-attractor-renderer/strange_attractor_renderer/).
//!
//! # Pipeline
//!
//! First you need to get a [`Config`].
//! I suggest creating it like this:
//!
//! ```
//! # use strange_attractor_renderer::*;
//! let config = Config {
//!     iterations: 100_000_000,
//!     ..Config::poisson_saturne()
//! };
//! ```
//!
//! ## Multithreaded
//!
//! Call [`render_parallel`].
//!
//! This benefits the most when the number of iterations is much higher than the dimensions of the
//! image. The gap closes rapidly when the relation is < 25. This also consumes more memory, as we
//! need a set of working images for each execution unit, usually the number of threads on your
//! CPU. See [Performance](#performance) for more info.
//!
//! ## Single-threaded
//!
//! Create a [`Runtime`].
//! Then [`render`] and finally [`colorize`].
//!
//! # Performance
//!
//! The thing slowing the algorithm down with larger image dimensions
//! is the cache size - and memory access. We basically do random access reads and writes on a
//! often > 2 megapixel image. If the system memory is slow, this brings performance to a halt.
//!
//! # Colouring
//!
//! When the iterations are executed, the magnitude of change is stored in a texture. When it's
//! time for colouring, this |Δp| is mapped to a palette. The brightness is determined by the number
//! of visits to the pixel.
// there are many #[allow()] in the code. These disregard the lint warnings I've enabled below.

// some online documentation wizardry
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
// deny all lints for the whole project, even the pedantic ones... :(
#![deny(clippy::all, clippy::pedantic)]
// allow the inline_always lint for the whole project, as they are heavily used to increase
// performance.
#![allow(clippy::inline_always)]

use std::fmt::Debug;
// import standard library items
use std::mem;
use std::sync::atomic::AtomicUsize;
use std::sync::{atomic, mpsc, Arc, Mutex};
use std::thread::JoinHandle;

// import items from external libraries
// image handles image formats and provide the main image type we use
use image::{GenericImage, GenericImageView, ImageBuffer, Luma, Pixel, Rgb, Rgba};
// used to get random initial points.
use rand::{Rng, SeedableRng};

pub use config::{ColorTransform, Colors, Config, RenderKind, View};
pub use primitives::{EulerAxisRotation, FloatExt, Vec3};

/// A strange attractor.
///
/// This is made generic to allow for speedy execution of all kinds of attractors.
pub trait Attractor: Debug + Clone {
    /// Get the next point of the attractor.
    ///
    /// Please make this `#[inline(always)]`!
    #[must_use]
    fn next_point(&self, previous: Vec3) -> Vec3;
}

/// Mathematical primitives
pub mod primitives {
    use super::Rng;
    use std::ops::{Add, Index, Mul, Sub};

    /// Trait to use the following functions on the float primitives.
    ///
    /// Here for convenience.
    pub trait FloatExt {
        #[must_use]
        fn square(self) -> Self;
        #[must_use]
        fn lerp(self, other: Self, t: Self) -> Self;
    }
    impl FloatExt for f64 {
        #[inline(always)]
        fn square(self) -> Self {
            self * self
        }
        #[inline(always)]
        fn lerp(self, other: Self, t: Self) -> Self {
            self * t + other * (1. - t)
        }
    }
    impl FloatExt for f32 {
        #[inline(always)]
        fn square(self) -> Self {
            self * self
        }
        #[inline(always)]
        fn lerp(self, other: Self, t: Self) -> Self {
            self * t + other * (1. - t)
        }
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    pub struct Vec3 {
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }
    impl Vec3 {
        #[must_use]
        pub fn new(x: f64, y: f64, z: f64) -> Self {
            Self { x, y, z }
        }

        /// Length of this vector.
        #[must_use]
        #[inline(always)]
        pub fn magnitude(self) -> f64 {
            (self.x.square() + self.y.square() + self.z.square()).sqrt()
        }
        #[must_use]
        #[inline(always)]
        pub fn normalize(self) -> Self {
            self * (1. / self.magnitude())
        }
    }
    // implement operations for Vec3
    impl Sub for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self::Output {
            Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
        }
    }
    impl Add for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self::Output {
            Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
        }
    }
    impl Mul<f64> for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, v: f64) -> Self::Output {
            Self::new(self.x * v, self.y * v, self.z * v)
        }
    }
    /// enables using `rng.gen` to get a Vec3, with x,y,z between 0 and 1
    impl rand::distributions::Distribution<Vec3> for rand::distributions::Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec3 {
            Vec3::new(rng.gen(), rng.gen(), rng.gen())
        }
    }

    /// See [Wikipedia](https://en.wikipedia.org/wiki/Euler's_rotation_theorem) for more info.
    #[derive(Debug, PartialEq, Clone, Copy)]
    pub struct EulerAxisRotation {
        /// The Euler axis
        pub axis: Vec3,
        /// Rotation around [`Self::axis`], in radians.
        pub rotation: f64,
    }
    impl EulerAxisRotation {
        #[must_use]
        #[inline(always)]
        pub fn to_rotation_matrix(self) -> Matrix3x3 {
            let Self { axis, rotation } = self;
            // normalize Vec, only on non-release/production builds, as this lowers performance
            #[cfg(debug_assertions)]
            let axis = axis.normalize();
            let Vec3 { x, y, z } = axis;
            let c = rotation.cos();
            let c1 = 1. - c;
            let s = rotation.sin();
            Matrix3x3 {
                columns: [
                    [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
                    [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
                    [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
                ],
            }
        }
    }

    /// A 3x3 matrix. Use `matrix[(0,2)]` to get the third item in the first column - indices are
    /// zero-based.
    #[derive(Debug, PartialEq, Clone)]
    pub struct Matrix3x3 {
        /// Each column contains a row with three [`f64`]s.
        pub columns: [[f64; 3]; 3],
    }
    impl Matrix3x3 {
        #[must_use]
        #[inline(always)]
        pub fn mul_right(&self, vec: Vec3) -> Vec3 {
            let m = self.columns;
            Vec3 {
                x: m[0][0] * vec.x + m[0][1] * vec.y + m[0][2] * vec.z,
                y: m[1][0] * vec.x + m[1][1] * vec.y + m[1][2] * vec.z,
                z: m[2][0] * vec.x + m[2][1] * vec.y + m[2][2] * vec.z,
            }
        }
    }
    impl Index<(usize, usize)> for Matrix3x3 {
        type Output = f64;
        #[inline(always)]
        fn index(&self, index: (usize, usize)) -> &Self::Output {
            &self.columns[index.0][index.1]
        }
    }
}

/// Configuration for rendering - how to get the next iteration, colouring, camera position,
/// brightness, etc.
pub mod config {
    use super::{attractors, Attractor, EulerAxisRotation, Rgb, Vec3};
    use std::fmt::Debug;

    /// How to render the internal data.
    #[derive(Debug, Clone)]
    pub enum RenderKind {
        /// The default, creates a good-looking gas-like image.
        Gas,
        /// Renders the depth map.
        Depth,
    }

    pub trait ColorTransform: Clone + Send + Sync + 'static {
        /// Please set the attribute `#[inline(always)]`.
        fn transform(&self, delta: Vec3, screen_space: Vec3, view: &View) -> f64;
    }
    impl<F: Fn(Vec3, Vec3, &View) -> f64 + Clone + Send + Sync + 'static> ColorTransform for F {
        fn transform(&self, delta: Vec3, screen_space: Vec3, view: &View) -> f64 {
            self(delta, screen_space, view)
        }
    }

    /// Other data which is dependant on the [`Attractor`].
    #[derive(Clone, Debug)]
    pub struct View {
        /// The position to center the camera on
        ///
        /// Is highly related to which coefficients are chosen
        pub center_camera: Vec3,
        pub rotation: EulerAxisRotation,
        /// General viewing scale. Increase this to zoom in more.
        pub scale: f64,
    }

    #[derive(Clone, Debug)]
    #[must_use]
    pub struct Config<A: Attractor, T: ColorTransform> {
        /// Heavily affects performance
        pub iterations: usize,
        /// Image width, slight performance decrease
        pub width: u32,
        /// Image height, slight performance decrease
        pub height: u32,

        pub render: RenderKind,
        /// This reduces colour quality.
        pub transparent: bool,
        /// The camera rotation angle.
        pub angle: f64,

        /// Be less verbose.
        pub silent: bool,

        pub attractor: A,
        pub colors: Colors,

        pub view: View,
        pub color_transform: T,
    }
    impl<A: Attractor, T: ColorTransform> Config<A, T> {
        pub fn new(coefficients: A, view: View, transform_colors: T) -> Self {
            Self {
                iterations: 10_000_000,
                width: 1920,
                height: 1080,

                render: RenderKind::Gas,
                transparent: true,
                angle: 0.0,

                silent: true,

                attractor: coefficients,
                colors: Colors::default(),

                view,
                color_transform: transform_colors,
            }
        }
    }
    impl Config<attractors::PolynomialSprott2Degree, color_transforms::Function> {
        pub fn poisson_saturne() -> Self {
            let coeffs = attractors::PolynomialSprott2Degree {
                x: [
                    0.021, 1.182, -1.183, 0.128, -1.12, -0.641, -1.152, -0.834, -0.97, 0.722,
                ],
                y: [
                    0.243_038, -0.825, -1.2, -0.835_443, -0.835_443, -0.364_557, 0.458, 0.622_785,
                    -0.394_937, -1.032_911,
                ],
                z: [
                    -0.455_696, 0.673, 0.915, -0.258_228, -0.495, -0.264, -0.432, -0.416, -0.877,
                    -0.3,
                ],
            };

            let view = View {
                /*
                    `TODO`: Add option to make first-pass to get these values, to then compute `center_camera`

                    Attractor size of the above

                    xmin = -0.327770, xmax = 0.335278  width 0.66
                    ymin = -0.012949, ymax = 0.492107  width 0.50
                    zmin = -0.628829, zmax = 0.103010  width 0.73
                */
                center_camera: Vec3::new(
                    -0.005,
                    0.262,
                    /* mid point between z[min,max]. constant 0.12 works well, don't know why we need that */
                    -0.366 + 0.12,
                ),
                rotation: EulerAxisRotation {
                    axis: Vec3 {
                        x: 0.304_289_493_528_802,
                        y: 0.760_492_682_863_655,
                        z: 0.573_636_455_813_981,
                    },
                    rotation: 1.782_681_918_874_46,
                },
                scale: 1.,
            };
            Self::new(coeffs, view, color_transforms::poisson_saturne)
        }
    }
    impl Config<attractors::PolynomialSprott2Degree, color_transforms::AdjustedVelocity> {
        pub fn solar_sail() -> Self {
            let coeffs = attractors::PolynomialSprott2Degree {
                x: [
                    0.744_304, -0.546_835, 0.121_519, -0.653_165, 0.399, 0.379, 0.44, 1.014,
                    -0.805_063, 0.377,
                ],
                y: [
                    -0.683, 0.531_646, -0.04557, -1.2, -0.546_835, 0.091_139, 0.744_304,
                    -0.273_418, -0.349_367, -0.531_646,
                ],
                z: [
                    0.712, 0.744_304, -0.577_215, 0.966, 0.04557, 1.063_291, 0.01519, -0.425_316,
                    0.212_658, -0.01519,
                ],
            };
            let view = View {
                center_camera: Vec3::new(0.28, -0.12, 0.22),
                rotation: EulerAxisRotation {
                    axis: Vec3::new(0.02466, 0.4618, -0.54789),
                    rotation: 2.2195,
                },
                scale: 1.7,
            };
            Self::new(
                coeffs,
                view,
                color_transforms::AdjustedVelocity {
                    factor: -0.2,
                    offset: 0.8,
                },
            )
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct BrighnessConstants {
        /// adds this to the colour before multiplying [`Self::factor`].
        /// Can be used to add contrast.
        /// This is usually what you want.
        pub offset: f64,
        pub factor: f64,
    }
    impl Default for BrighnessConstants {
        fn default() -> Self {
            Self {
                offset: -0.15,
                factor: 5. / 3.,
            }
        }
    }

    #[derive(Debug, Clone)]
    #[must_use]
    pub struct Palette {
        list: Vec<Rgb<f64>>,
        count_f64: f64,
    }
    impl Palette {
        /// # Panics
        ///
        /// Panics if `list.is_empty()`.
        pub fn new(mut list: Vec<Rgb<f64>>) -> Self {
            list.reserve_exact(1);
            list.push(*list.last().unwrap());
            #[allow(clippy::cast_precision_loss)]
            Self {
                count_f64: (list.len() - 1) as _,
                list,
            }
        }
        pub fn from_rgb<const LEN: usize>(r: [f64; LEN], g: [f64; LEN], b: [f64; LEN]) -> Self {
            let mut colors = Vec::with_capacity(LEN + 1);
            for i in 0..LEN {
                colors.push(Rgb([r[i], g[i], b[i]]));
            }
            Self::new(colors)
        }
        /// Number of colours in this palette.
        #[inline(always)]
        #[must_use]
        pub fn count(&self) -> usize {
            self.list.len() - 1
        }
        /// `value` is position in color wheel, in the range [0..1)
        /// If `value` is out of that range, it's clamped.
        #[inline(always)]
        #[must_use]
        pub fn interpolate(&self, value: f64) -> Rgb<f64> {
            let value = if value < 0. {
                0.
            } else if value >= 1. {
                0.999_999
            } else {
                value
            };

            let value = value * self.count_f64;
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let n = (value.floor()) as usize;
            let sub_n_offset = value % 1.;
            let sub_n_offset_1 = 1.0 - sub_n_offset;

            // SAFETY: we asserted above that `value` is in an appropriate range for this.
            let [r1, g1, b1] = unsafe { self.list.get_unchecked(n).0 };
            let [r2, g2, b2] = unsafe { self.list.get_unchecked(n + 1).0 };
            Rgb([
                // lerp between colours
                //
                // the lerp is inlined to avoid multiple subtractions, about 3% performance increase
                // (r[n + 1].lerp(r[n], sub_n_offset)).sqrt(),
                // (g[n + 1].lerp(g[n], sub_n_offset)).sqrt(),
                // (b[n + 1].lerp(b[n], sub_n_offset)).sqrt(),
                //
                (r2 * sub_n_offset + r1 * sub_n_offset_1).sqrt(),
                (g2 * sub_n_offset + g1 * sub_n_offset_1).sqrt(),
                (b2 * sub_n_offset + b1 * sub_n_offset_1).sqrt(),
            ])
        }
    }
    #[derive(Debug, Clone)]
    pub struct Colors {
        pub palette: Palette,

        pub brighness: BrighnessConstants,
    }
    impl Default for Colors {
        fn default() -> Self {
            Self {
                palette: Palette::from_rgb(
                    [1., 0.5, 1., 0.5, 0.5, 1.],
                    [1., 1., 0.5, 1., 0.5, 0.5],
                    [0.5, 0.5, 0.5, 1., 1., 1.],
                ),

                brighness: BrighnessConstants::default(),
            }
        }
    }

    /// Transformations for getting the position on the palette used in colouring.
    /// Returned values should range between [0..1).
    /// All functions used as [colour transforms](Config::color_transform) must take three
    /// arguments - the Δp, the position in screen space, and the [`View`].
    pub mod color_transforms {
        use super::{ColorTransform, Vec3, View};
        use std::f64::consts::PI;

        /// The raw function transform.
        pub type Function = fn(Vec3, Vec3, &View) -> f64;

        /// Calculated as `(delta.magnitude() + offset) * factor`.
        #[derive(Clone, Debug)]
        pub struct AdjustedVelocity {
            pub offset: f64,
            pub factor: f64,
        }
        impl ColorTransform for AdjustedVelocity {
            #[inline(always)]
            fn transform(&self, delta: Vec3, _screen_space: Vec3, _view: &View) -> f64 {
                (delta.magnitude() + self.offset) * self.factor
            }
        }
        #[must_use]
        #[inline(always)]
        pub fn poisson_saturne(delta: Vec3, screen_space: Vec3, coeffs: &View) -> f64 {
            #[inline(always)]
            fn part(p: Vec3, coeffs: &View) -> f64 {
                #[allow(unused)] // clarity
                const RADIAN_45_5: f64 = 91. * PI / 360.;
                /// [`RADIAN_45_5`].cos()
                ///
                /// This was calculated using Rust on x64 Linux.
                #[allow(clippy::excessive_precision)]
                const COS: f64 =
                    0.700_909_264_299_850_898_183_308_345_323_894_172_906_875_610_351_562_5;
                /// [`RADIAN_45_5`].sin()
                ///
                /// This was calculated using Rust on x64 Linux.
                #[allow(clippy::excessive_precision)]
                const SIN: f64 =
                    0.713_250_449_154_181_564_992_427_411_198_150_366_544_723_510_742_187_5;

                let x2 =
                    (p.x + coeffs.center_camera.x) * COS + (p.z + coeffs.center_camera.y) * SIN;
                // This computes which "part" of the poisson saturne attractor the current point is
                // in (in screen space) by comparing the points to limiting "planes"
                if x2 < -0.0839
                    || 10.55 * x2 + p.y < 0.46 - 1.0941
                    || 1.0426 * x2 + p.y < 0.179 - 0.1576
                    || 0.5139 * x2 - p.y > -0.04 - 0.04092
                {
                    0.
                } else {
                    1.
                }
            }
            // Using the part here means we shift half the palette (it's divided by two).
            // To not make the returned value return more than 1, we adjust the |Δp| (magnitude of
            // point delta). First by dividing by 2, then by subtracting the whole (with the part
            // data) by 0.1, then dividing by 0.9.
            let color = (part(screen_space, coeffs) + delta.magnitude()) / 2.;
            (color - 0.1) / 0.9
        }
    }
}

/// The included attractors.
///
/// > You can always implement [`Attractor`] yourself!
///
/// These are majorly inspired from [chaoscope](http://chaoscope.org/manual.htm)'s manual.
pub mod attractors {
    use super::{Attractor, FloatExt, Vec3};

    /// Coefficients for a polynomial Sprott type attractor, of the second degree.
    /// See this page from [chaoscope](http://www.chaoscope.org/doc/attractors.htm) for more
    /// context.
    #[derive(Debug, Clone)]
    #[must_use]
    pub struct PolynomialSprott2Degree {
        // coefficient lists
        pub x: [f64; 10],
        pub y: [f64; 10],
        pub z: [f64; 10],
    }
    /// This is part of the polynomial Sprott algorithm. We get the new coordinates by multiplying previous
    /// (using polynomials) with a set of coefficients.
    impl Attractor for PolynomialSprott2Degree {
        #[inline(always)]
        fn next_point(&self, p: crate::Vec3) -> crate::Vec3 {
            #[inline(always)]
            // polynomials is a reference to an array with 10 f64s.
            fn sum_coefficients(polynomials: &[f64; 10], coefficients: &[f64; 10]) -> f64 {
                let mut sum = 0.;
                for i in 0..10 {
                    // unsafe to circumvent bounds checks, increasing speed
                    // SAFETY: we know 0..10 is in bounds of the array of length 10 (i.e. [f64; 10])
                    unsafe {
                        let v1 = polynomials.get_unchecked(i);
                        let v2 = coefficients.get_unchecked(i);
                        sum += v1 * v2;
                    }
                }
                sum
            }
            // monomial (polynomials with only one term)
            let monoms = [
                1.,
                p.x,
                p.x.square(),
                p.x * p.y,
                p.x * p.z,
                p.y,
                p.y.square(),
                p.y * p.z,
                p.z,
                p.z.square(),
            ];

            Vec3 {
                x: sum_coefficients(&monoms, &self.x),
                y: sum_coefficients(&monoms, &self.y),
                z: sum_coefficients(&monoms, &self.z),
            }
        }
    }
}

/// Convenience alias for an 16-bit RGBA image.
pub type FinalImage = ImageBuffer<Rgba<u16>, Vec<u16>>;

/// Stores data used by the algorithm.
///
/// This enables us to reuse memory.
#[must_use]
pub struct Runtime {
    // counts the number of visits to each pixel
    count: ImageBuffer<Luma<u32>, Vec<u32>>,
    // counts the result of [`config::transforms`]
    steps: ImageBuffer<Luma<f64>, Vec<f64>>,
    // stores the depth of each pixel (in screen space)
    // the range pixel values aren't defined, iterate the whole list to get the minimum and
    // maximum.
    zbuf: ImageBuffer<Luma<f32>, Vec<f32>>,
    // the max number of steps
    // used to scale all other pixels in [`Self::steps`], without having to iterate the whole
    // texture first.
    max: u32,

    rng: rand::rngs::SmallRng,
}
impl Runtime {
    /// You have to [`Self::set_width_height`] before using this.
    fn empty() -> Self {
        Self {
            count: ImageBuffer::new(0, 0),
            steps: ImageBuffer::new(0, 0),
            zbuf: ImageBuffer::new(0, 0),
            max: 0,

            rng: rand::rngs::SmallRng::from_entropy(),
        }
    }
    /// Creates a new runtime from the dimensions of [`Config`].
    pub fn new(config: &Config<impl Attractor, impl ColorTransform>) -> Self {
        let mut me = Self::empty();
        me.set_width_height(config.width, config.height);
        me.reset();
        me
    }
    /// Makes new textures if the width and height don't match the inner textures.
    fn set_width_height(&mut self, width: u32, height: u32) {
        if self.count.width() != width || self.count.height() != height {
            self.count = ImageBuffer::new(width, height);
            self.steps = ImageBuffer::new(width, height);
            self.zbuf = ImageBuffer::new(width, height);
            self.max = 0;
            self.reset();
        }
    }
    /// Returns an empty image of any pixel type.
    fn image_identity<T: Pixel>() -> ImageBuffer<T, Vec<T::Subpixel>> {
        ImageBuffer::from_raw(0, 0, Vec::new()).unwrap()
    }
    /// Reset this runtime.
    #[allow(clippy::missing_panics_doc)] // doesn't happen
    pub fn reset(&mut self) {
        let width = self.count.width();
        let height = self.count.height();
        let count = mem::replace(&mut self.count, Self::image_identity());
        let mut count = count.into_raw();
        count.fill(0);
        let steps = mem::replace(&mut self.steps, Self::image_identity());
        let mut steps = steps.into_raw();
        steps.fill(0.);
        let zbuf = mem::replace(&mut self.zbuf, Self::image_identity());
        let mut zbuf = zbuf.into_raw();
        zbuf.fill(-1.);
        self.max = 0;

        self.count = ImageBuffer::from_raw(width, height, count).unwrap();
        self.steps = ImageBuffer::from_raw(width, height, steps).unwrap();
        self.zbuf = ImageBuffer::from_raw(width, height, zbuf).unwrap();
    }

    /// Merges the data of the two images. `other` will not be modified.
    /// This makes `self` appear as if it had been rendered with the
    /// sum of the iterations of `self` and `other`.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of `self` isn't the same as the dimensions of `other`.
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(self.steps.width(), other.steps.width());
        assert_eq!(self.steps.height(), other.steps.height());

        for x in 0..(self.steps.width()) {
            for y in 0..(self.steps.height()) {
                unsafe {
                    let mut pixel = self.count.unsafe_get_pixel(x, y);
                    let other_pixel = other.count.unsafe_get_pixel(x, y);
                    // `.0[0]` since the pixel is wrapped in a `Luma`, which contains a unnamed slice (`.0`),
                    // which we get the first and only element of (`[0]`)
                    pixel.0[0] += other_pixel.0[0];
                    self.count.unsafe_put_pixel(x, y, pixel);
                    if pixel.0[0] > self.max {
                        self.max = pixel.0[0];
                    }
                };

                let zbuf_pix1 = unsafe { self.zbuf.unsafe_get_pixel(x, y) };
                let zbuf_pix2 = unsafe { other.zbuf.unsafe_get_pixel(x, y) };
                if zbuf_pix2.0[0] > zbuf_pix1.0[0] {
                    unsafe {
                        let other_step = other.steps.unsafe_get_pixel(x, y);
                        self.steps.unsafe_put_pixel(x, y, other_step);

                        self.zbuf.unsafe_put_pixel(x, y, zbuf_pix2);
                    }
                }
            }
        }
    }
}
/// Render according to `config`, with angle `rotation` around the attractor.
///
/// If the [`Runtime`] isn't [cleared](Runtime::reset), this just continues the "building" of the
/// image. This can therefore be called in succession and the result is an ever-improving image.
///
/// `rotation` is around [`View::center_camera`], in radians.
#[allow(clippy::many_single_char_names)]
pub fn render(config: &Config<impl Attractor, impl ColorTransform>, runtime: &mut Runtime) {
    let mut initial_point = runtime.rng.gen::<Vec3>() * 0.1;
    // skip first 1000 to get good values in the attractor
    for _ in 0..1000 {
        initial_point = config.attractor.next_point(initial_point);
    }

    // computations used later - we do as much work up front as possible
    let rotation_matrix = config.view.rotation.to_rotation_matrix();
    let sin_v = config.angle.sin();
    let cos_v = config.angle.cos();
    let center_camera = config.view.center_camera;
    #[allow(clippy::cast_lossless)]
    let width = config.width as f64;
    #[allow(clippy::cast_lossless)]
    let height = config.height as f64;
    let width_scaled = width * config.view.scale;
    let scale_adjusted_mid = 0.5 / config.view.scale;

    let mut previous_point = initial_point;
    let mut current_point = initial_point;

    for _ in 0..(config.iterations) {
        current_point = config.attractor.next_point(current_point);

        // rotation_matrix * current_point
        let screen_space = rotation_matrix.mul_right(current_point);

        // rotate around center_camera
        let x2 =
            (screen_space.x + center_camera.x) * cos_v + (screen_space.z + center_camera.y) * sin_v;
        let z2 =
            (screen_space.x + center_camera.x) * sin_v - (screen_space.z + center_camera.y) * cos_v;

        // (0.5 - x2 * scale) * width, but optimized to use constants, so we don't have the
        // multiplication
        let i = (scale_adjusted_mid - x2) * width_scaled;
        // instead of 0.5width as above, we have 0.5height as the center point. The scaling of the
        // position relative to that is still width, as we want to keep the shape
        let j = height / 2. - (screen_space.y + center_camera.z) * width_scaled;

        // do bounds checks
        if i >= width || j >= height || i < 0. || j < 0. {
            // this is incredibly important:
            // if we don't set the previous point, if we don't follow the path, even if the point
            // is outside the viewport, the delta is used in colouring.
            previous_point = current_point;
            continue;
        }

        // convert floats to image coordinates
        // this is safe, as we checked the bounds above.
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let i = i as u32;
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let j = j as u32;

        // unsafe because we use unsafe functions to get pixel values without checking bounds of
        // inner slice. We know the values are within bounds, as by the if ... { continue; } block
        // above.
        unsafe {
            let mut pixel = runtime.count.unsafe_get_pixel(i, j);
            // `.0[0]` since the pixel is wrapped in a `Luma`, which contains a unnamed slice (`.0`),
            // which we get the first and only element of (`[0]`)
            pixel.0[0] += 1;
            runtime.count.unsafe_put_pixel(i, j, pixel);
            if pixel.0[0] > runtime.max {
                runtime.max = pixel.0[0];
            }
        };

        let zbuf_pix = unsafe { runtime.zbuf.unsafe_get_pixel(i, j) };
        #[allow(clippy::cast_possible_truncation)]
        // if new depth is greater than previous, change that pixel
        if z2 as f32 > zbuf_pix.0[0] {
            let delta = current_point - previous_point;

            // get the colour transformation output, later used in colouring as the index to the
            // palette
            let value = config
                .color_transform
                .transform(delta, screen_space, &config.view);
            unsafe {
                runtime.steps.unsafe_put_pixel(i, j, Luma([value]));

                runtime.zbuf.unsafe_put_pixel(i, j, Luma([z2 as f32]));
            }
        }

        previous_point = current_point;
    }
}
#[must_use]
#[allow(clippy::missing_panics_doc)]
pub fn colorize(
    config: &Config<impl Attractor, impl ColorTransform>,
    runtime: &Runtime,
) -> FinalImage {
    let bk = config.colors.brighness;
    let mut image = ImageBuffer::new(config.width, config.height);

    let u16_max = f64::from(u16::MAX);

    // ignore lints
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    match config.render {
        RenderKind::Gas => {
            for ((x, y, steps), count) in
                runtime.steps.enumerate_pixels().zip(runtime.count.pixels())
            {
                let color = config.colors.palette.interpolate(steps.0[0]);
                let [r, g, b] = color.0;
                // add 1 to both to not get any logs of values under 1.
                let factor = f64::from(count.0[0] + 1).log(f64::from(runtime.max + 1));
                let pixel = Rgba([
                    ((r * factor + bk.offset) * bk.factor * u16_max) as _,
                    ((g * factor + bk.offset) * bk.factor * u16_max) as _,
                    ((b * factor + bk.offset) * bk.factor * u16_max) as _,
                    if config.transparent {
                        (factor * u16_max) as u16
                    } else {
                        u16::MAX
                    },
                ]);
                // safety: `image` has the same size as all the others
                unsafe { image.unsafe_put_pixel(x, y, pixel) };
            }
        }
        RenderKind::Depth => {
            #[allow(clippy::float_cmp)] // the -1.0 value is set by us
            let (max, min) = runtime
                .zbuf
                .pixels()
                .map(|pixel| pixel.0[0])
                .filter(|p| *p != -1.0)
                .fold((0.0f32, f32::MAX), |(a1, a2), p| (a1.max(p), a2.min(p)));
            let diff = max - min;

            for (x, y, z) in runtime.zbuf.enumerate_pixels() {
                let z = z.0[0];
                #[allow(clippy::float_cmp)]
                // if z hasn't changed from default, we return 0.
                let z = if z == -1.0 {
                    0.0
                } else {
                    // reverse lerp
                    (z - min) / diff
                };
                let z = (z * f32::from(u16::MAX)) as u16;
                let pixel = Rgba([z, z, z, u16::MAX]);
                // safety: `image` has the same size as all the others
                unsafe { image.unsafe_put_pixel(x, y, pixel) };
            }
        }
    }

    image
}

/// Handle to threads and channels to render a config on multiple threads.
#[must_use]
pub struct ParallelRenderer<A: Attractor, T: ColorTransform> {
    threads: Vec<JoinHandle<()>>,
    render_receiver: mpsc::Receiver<Arc<Mutex<Runtime>>>,
    // `TODO`: make the config we send dynamic and downcast, so we don't have to have this
    // generic.
    #[allow(clippy::type_complexity)]
    job_sender: watch::WatchSender<Option<(Config<A, T>, Arc<AtomicUsize>)>>,
}
impl<A: Attractor + Send + Sync + 'static, T: ColorTransform> ParallelRenderer<A, T> {
    /// Initiate an appropriate amount of threads and set them up to accept jobs.
    #[allow(clippy::missing_panics_doc)]
    pub fn new() -> Self {
        let num_threads = std::thread::available_parallelism()
            .unwrap_or(std::num::NonZeroUsize::new(8).unwrap())
            .get();

        let mut threads = Vec::with_capacity(num_threads);

        let (job_sender, mut job_receiver) = watch::channel(None);
        // get the first, `None` value.
        job_receiver.wait();
        let (render_sender, render_receiver) = mpsc::channel();

        for _ in 0..num_threads {
            let receiver = job_receiver.clone();
            let sender = render_sender.clone();
            let handle = std::thread::spawn(move || {
                let mut receiver = receiver;
                let sender = sender;

                let runtime = Arc::new(Mutex::new(Runtime::empty()));

                loop {
                    let (config, job_counter): (Config<_, _>, Arc<AtomicUsize>) =
                        if let Some(m) = receiver.wait() {
                            m
                        } else {
                            return;
                        };

                    {
                        let mut rt = runtime.lock().unwrap();
                        rt.set_width_height(config.width, config.height);
                        rt.reset();

                        if !config.silent {
                            println!("Rendering started on thread.");
                        }
                        loop {
                            // apply the function to check if we should continue.
                            // This also updates the atomic value.
                            //
                            // A `updated` valuable is needed to only decrement once.
                            let mut updated = false;
                            let more_to_do = job_counter
                                .fetch_update(
                                    atomic::Ordering::SeqCst,
                                    atomic::Ordering::SeqCst,
                                    |v| {
                                        if v > 0 {
                                            if updated {
                                                Some(v)
                                            } else {
                                                if !config.silent {
                                                    println!("Iteration complete, {v} left to go.");
                                                }
                                                updated = false;
                                                Some(v - 1)
                                            }
                                        } else {
                                            None
                                        }
                                    },
                                )
                                .is_ok();

                            if !more_to_do {
                                break;
                            }
                            render(&config, &mut rt);
                        }
                    }
                    sender.send(Arc::clone(&runtime)).unwrap();
                    if !config.silent {
                        println!("Rendered finished.");
                    }
                }
            });
            threads.push(handle);
        }
        println!("Created parallel renderer.");
        Self {
            threads,
            render_receiver,
            job_sender,
        }
    }
    /// Send the `job` to all threads.
    fn send(&mut self, job: Config<A, T>, job_counter: Arc<AtomicUsize>) {
        self.job_sender.send(Some((job, job_counter)));
    }
    /// Blocks on receiving access to the thread's runtimes.
    /// Access them though the [`Mutex`].
    fn recv(&mut self) -> impl Iterator<Item = Arc<Mutex<Runtime>>> + '_ {
        self.render_receiver.iter().take(self.num_threads())
    }
    /// Number of initiated threads, usually used to construct the `job_counter` at [`Self::send`].
    fn num_threads(&self) -> usize {
        self.threads.len()
    }
    /// Wait for all threads to finish.
    pub fn shutdown(self) {
        self.job_sender.send(None);
        self.threads
            .into_iter()
            .for_each(|thread| thread.join().expect("render thread panicked"));
    }
}
impl<A: Attractor + Send + Sync + 'static, T: ColorTransform> Default for ParallelRenderer<A, T> {
    fn default() -> Self {
        Self::new()
    }
}
/// I recommend `16` for `jobs_per_thread`. If you get uneven images with low iteration counts, try
/// `8`.
///
/// At relatively low rations between iterations and pixels (<50), this isn't much faster.
///
/// # How it works
///
/// Because strange attractors are inherently chaotic, we can render multiply images and then add
/// them together.
/// Internally, this uses three textures, `count` for the number of visits to the pixel, `steps`
/// for the velocity at the closest visitation, and the `zbuf` which can give information about
/// which pixel is the closest. The `zbuf` also gives us the depth texture.
/// When combining the rendered image, we have to combine all of these.
///
/// When rendering through [`render`] without resetting the [`Runtime`], this is what naturally
/// happens. When we use multiple threads however, we have to explicitly combine the runtimes
/// consistently to what the [`render`] method implicitly does.
#[allow(clippy::missing_panics_doc)] // it won't panic
#[must_use]
pub fn render_parallel<A: Attractor + Send + Sync + 'static, T: ColorTransform>(
    renderer: &mut ParallelRenderer<A, T>,
    mut config: Config<A, T>,
    jobs_per_thread: usize,
) -> FinalImage {
    let iterations = config.iterations;
    // split up in num_threads and jobs_per_thread
    config.iterations = iterations / renderer.num_threads() / jobs_per_thread;

    // We keep a job counter to balance threads. If one is way slower, it'll "take" less jobs,
    // which results in better runtime.
    let job_counter = Arc::new(AtomicUsize::new(jobs_per_thread * renderer.num_threads()));

    // send the job
    renderer.send(config.clone(), job_counter);

    // wait for the job
    let mut iter = renderer.recv();
    // UNWRAP: `available_parallelism` is guaranteed to always return >0
    let current = iter.next().unwrap();
    // merge all images
    for runtime in iter {
        let mut a = current.lock().unwrap();
        let b = runtime.lock().unwrap();
        a.merge(&b);
    }

    {
        let current = current.lock().unwrap();
        colorize(&config, &current)
    }
}
