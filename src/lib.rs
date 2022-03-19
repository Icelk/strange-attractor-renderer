//! # Pipeline
//!
//! First you need to get a [`Config`].
//! I suggest creating it like this:
//!
//! ```
//! # use strange_attractor_renderer::*;
//! let config = Config {
//!     iterations: 100_000_000,
//!     ..Default::default()
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

#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::inline_always)]

use std::mem;
use std::sync::atomic::AtomicUsize;
use std::sync::{atomic, mpsc, Arc, Mutex};
use std::thread::JoinHandle;

use image::{GenericImage, GenericImageView, ImageBuffer, Luma, Pixel, Rgb, Rgba};
use rand::{Rng, SeedableRng};

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

pub use config::Config;
use config::{CoefficientList, Coefficients, Colors, RenderKind};
use primitives::{EulerAxisRotation, Vec3};

pub mod primitives {
    use super::{FloatExt, Rng};

    use std::ops::{Add, Index, Mul, Sub};
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
    }
    impl Sub for Vec3 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self::Output {
            Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
        }
    }
    impl Add for Vec3 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
        }
    }
    impl Mul<f64> for Vec3 {
        type Output = Self;
        fn mul(self, v: f64) -> Self::Output {
            Self::new(self.x * v, self.y * v, self.z * v)
        }
    }
    impl rand::distributions::Distribution<Vec3> for rand::distributions::Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec3 {
            Vec3::new(rng.gen(), rng.gen(), rng.gen())
        }
    }
    #[derive(Debug, PartialEq, Clone, Copy)]
    pub struct EulerAxisRotation {
        pub axis: Vec3,
        /// Rotation around [`Self::axis`], in radians.
        pub rotation: f64,
    }
    impl EulerAxisRotation {
        #[must_use]
        #[inline(always)]
        pub fn to_rotation_matrix(self) -> Matrix3x3 {
            let Self { axis, rotation } = self;
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

    #[derive(Debug, PartialEq, Clone)]
    pub struct Matrix3x3 {
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

pub mod config {
    use std::fmt::{self, Debug};

    use super::{EulerAxisRotation, Vec3};

    #[derive(Debug, Clone)]
    pub enum RenderKind {
        Gas,
        Depth,
    }

    #[derive(Debug, Clone)]
    pub struct Config {
        pub iterations: usize,
        pub width: u32,
        pub height: u32,

        pub render: RenderKind,
        pub transparent: bool,
        pub angle: f64,

        pub silent: bool,

        pub coefficients: Coefficients,
        pub rotation: EulerAxisRotation,
        pub colors: Colors,
    }
    impl Default for Config {
        fn default() -> Self {
            Self {
                iterations: 10_000_000,
                width: 1920,
                height: 1080,

                render: RenderKind::Gas,
                transparent: true,
                angle: 0.0,

                silent: true,

                coefficients: Coefficients::default(),
                rotation: EulerAxisRotation {
                    axis: Vec3 {
                        x: 0.304_289_493_528_802,
                        y: 0.760_492_682_863_655,
                        z: 0.573_636_455_813_981,
                    },
                    rotation: 1.782_681_918_874_46,
                },
                colors: Colors::default(),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct CoefficientList {
        pub list: [f64; 10],
    }
    #[derive(Clone)]
    pub struct Coefficients {
        pub x: CoefficientList,
        pub y: CoefficientList,
        pub z: CoefficientList,

        /// The position to center the camera on
        ///
        /// Is highly related to which coefficients are chosen
        pub center_camera: Vec3,

        /// Takes delta, screen space, and settings.
        pub transform_colors: fn(Vec3, Vec3, &Self) -> f64,
    }
    impl Debug for Coefficients {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Coeffients")
                .field("x", &self.x)
                .field("y", &self.y)
                .field("z", &self.z)
                .field("center_camera", &self.center_camera)
                .field("transform_colors", &"fn")
                .finish()
        }
    }
    impl Default for Coefficients {
        fn default() -> Self {
            Self {
                x: CoefficientList {
                    list: [
                        0.021, 1.182, -1.183, 0.128, -1.12, -0.641, -1.152, -0.834, -0.97, 0.722,
                    ],
                },
                y: CoefficientList {
                    list: [
                        0.243_038, -0.825, -1.2, -0.835_443, -0.835_443, -0.364_557, 0.458,
                        0.622_785, -0.394_937, -1.032_911,
                    ],
                },
                z: CoefficientList {
                    list: [
                        -0.455_696, 0.673, 0.915, -0.258_228, -0.495, -0.264, -0.432, -0.416,
                        -0.877, -0.3,
                    ],
                },

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
                transform_colors: defaults::color_transform,
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct BrighnessConstants {
        /// adds this to the colour before multiplying [`Self::factor`].
        /// Can be used to add contrast.
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

    /// Each of the slices should have their last and second to last be the same.
    #[derive(Debug, Clone)]
    pub struct Colors {
        pub r: [f64; 7],
        pub g: [f64; 7],
        pub b: [f64; 7],

        pub brighness: BrighnessConstants,
    }
    impl Default for Colors {
        fn default() -> Self {
            Self {
                r: [1., 0.5, 1., 0.5, 0.5, 1., 1.],
                g: [1., 1., 0.5, 1., 0.5, 0.5, 0.5],
                b: [0.5, 0.5, 0.5, 1., 1., 1., 1.],

                brighness: BrighnessConstants::default(),
            }
        }
    }

    pub mod defaults {
        use super::{Coefficients, Vec3};
        use std::f64::consts::PI;

        #[must_use]
        #[inline(always)]
        pub fn color_transform(delta: Vec3, screen_space: Vec3, coeffs: &Coefficients) -> f64 {
            #[inline(always)]
            fn part(p: Vec3, coeffs: &Coefficients) -> f64 {
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
            // `TODO`: implement "brighten"
            let color = (part(screen_space, coeffs) + delta.magnitude()) / 2.;
            (color - 0.1) / 0.9
        }
    }
}

pub type FinalImage = ImageBuffer<Rgba<u16>, Vec<u16>>;

/// This is part of the general algorithm. We get the new coordinates by multiplying previous
/// (using polynomials) with a set of coefficients.
#[inline(always)]
#[must_use]
pub fn next_point(p: Vec3, coefficients: &Coefficients) -> Vec3 {
    #[inline(always)]
    fn sum_coefficients(polynomials: &[f64; 10], coefficients: &CoefficientList) -> f64 {
        let mut sum = 0.;
        for i in 0..10 {
            unsafe {
                let v1 = polynomials.get_unchecked(i);
                let v2 = coefficients.list.get_unchecked(i);
                sum += v1 * v2;
            }
        }
        sum
    }
    let polynomials = [
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
        x: sum_coefficients(&polynomials, &coefficients.x),
        y: sum_coefficients(&polynomials, &coefficients.y),
        z: sum_coefficients(&polynomials, &coefficients.z),
    }
}
/// `value` is position in color wheel, in the range [0..1)
#[must_use]
#[inline(always)]
pub fn color(value: f64, colors: &Colors) -> Rgb<f64> {
    let Colors {
        r,
        g,
        b,
        brighness: _,
    } = colors;
    let value = value * 6.;
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let n = (value.floor()) as usize;
    let sub_n_offset = value % 1.;
    let sub_n_offset_1 = 1.0 - sub_n_offset;
    Rgb([
        // lerp between colours
        //
        // the lerp is inlined to avoid multiple subtractions, about 3% performance increase
        // (r[n + 1].lerp(r[n], sub_n_offset)).sqrt(),
        // (g[n + 1].lerp(g[n], sub_n_offset)).sqrt(),
        // (b[n + 1].lerp(b[n], sub_n_offset)).sqrt(),
        //
        (r[n + 1] * sub_n_offset + r[n] * sub_n_offset_1).sqrt(),
        (g[n + 1] * sub_n_offset + g[n] * sub_n_offset_1).sqrt(),
        (b[n + 1] * sub_n_offset + b[n] * sub_n_offset_1).sqrt(),
    ])
}

/// Stores data used by the algorithm.
///
/// This enables us to reuse memory.
#[must_use]
pub struct Runtime {
    count: ImageBuffer<Luma<u32>, Vec<u32>>,
    steps: ImageBuffer<Luma<f64>, Vec<f64>>,
    zbuf: ImageBuffer<Luma<f32>, Vec<f32>>,
    max: u32,

    rng: rand::rngs::SmallRng,
}
impl Runtime {
    fn empty() -> Self {
        Self {
            count: ImageBuffer::new(0, 0),
            steps: ImageBuffer::new(0, 0),
            zbuf: ImageBuffer::new(0, 0),
            max: 0,

            rng: rand::rngs::SmallRng::from_entropy(),
        }
    }
    pub fn new(config: &Config) -> Self {
        let mut me = Self::empty();
        me.set_width_height(config.width, config.height);
        me.reset();
        me
    }
    fn set_width_height(&mut self, width: u32, height: u32) {
        if self.count.width() != width || self.count.height() != height {
            self.count = ImageBuffer::new(width, height);
            self.steps = ImageBuffer::new(width, height);
            self.zbuf = ImageBuffer::new(width, height);
            self.max = 0;
            self.reset();
        }
    }
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
/// If the [`Runtime`] isn't [cleared](Runtime::reset), this just continues the "building" of the
/// image. This can therefore be called in succession and the result is an ever-improving image.
///
/// `rotation` is around [`Coefficients::center_camera`], in radians.
#[allow(clippy::many_single_char_names)]
pub fn render(config: &Config, runtime: &mut Runtime) {
    let mut initial_point = runtime.rng.gen::<Vec3>() * 0.1;
    for _ in 0..1000 {
        initial_point = next_point(initial_point, &config.coefficients);
    }

    let rotation_matrix = config.rotation.to_rotation_matrix();
    let sin_v = config.angle.sin();
    let cos_v = config.angle.cos();
    let center_camera = config.coefficients.center_camera;
    #[allow(clippy::cast_lossless)]
    let width = config.width as f64;
    #[allow(clippy::cast_lossless)]
    let height = config.height as f64;

    let mut previous_point = initial_point;
    let mut current_point = initial_point;

    for _ in 0..(config.iterations) {
        current_point = next_point(current_point, &config.coefficients);

        let screen_space = rotation_matrix.mul_right(current_point);

        // rotate around center_camera
        let x2 =
            (screen_space.x + center_camera.x) * cos_v + (screen_space.z + center_camera.y) * sin_v;
        let z2 =
            (screen_space.x + center_camera.x) * sin_v - (screen_space.z + center_camera.y) * cos_v;

        let i = (0.5 - x2) * width;
        let j = height / 2. - (screen_space.y + center_camera.z) * width;

        if i >= width || j >= height || i < 0. || j < 0. {
            continue;
        }

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
        if z2 as f32 > zbuf_pix.0[0] {
            let delta = current_point - previous_point;

            let value =
                (config.coefficients.transform_colors)(delta, screen_space, &config.coefficients);
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
pub fn colorize(config: &Config, runtime: &Runtime) -> FinalImage {
    let bk = config.colors.brighness;
    let mut image = ImageBuffer::new(config.width, config.height);

    #[allow(clippy::cast_lossless)]
    let u16_max = u16::MAX as f64;

    let depth_info = if matches!(config.render, RenderKind::Depth) {
        Some(
            #[allow(clippy::float_cmp)] // the -1.0 value is set by us
            runtime
                .zbuf
                .pixels()
                .map(|pixel| pixel.0[0])
                .filter(|p| *p != -1.0)
                // have to use .fold instead of .max as `f32` doesn't implement Ord.
                .fold((0.0f32, f32::MAX), |(a1, a2), p| (a1.max(p), a2.min(p))),
        )
    } else {
        None
    };
    let depth_info = depth_info.map(|(a, b)| (a, b, (a - b)));

    // ignore lints
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless
    )]
    for ((x, y, steps), (count, z)) in runtime
        .steps
        .enumerate_pixels()
        .zip(runtime.count.pixels().zip(runtime.zbuf.pixels()))
    {
        let color = color(steps.0[0], &config.colors);
        let [r, g, b] = color.0;
        // add 1 to both to not get any logs of values under 1.
        let factor = ((count.0[0] + 1) as f64).log((runtime.max + 1) as f64);
        let pixel = match config.render {
            RenderKind::Gas => Rgba([
                ((r * factor + bk.offset) * bk.factor * u16_max) as _,
                ((g * factor + bk.offset) * bk.factor * u16_max) as _,
                ((b * factor + bk.offset) * bk.factor * u16_max) as _,
                if config.transparent {
                    (factor * u16_max) as u16
                } else {
                    u16::MAX
                },
            ]),
            RenderKind::Depth => {
                let z = z.0[0];
                #[allow(clippy::float_cmp)]
                // if z hasn't changed from default, we return 0.
                let z = if z == -1.0 {
                    0.0
                } else {
                    let (_, min, diff) = depth_info.unwrap();
                    (z - min) / diff
                };
                let z = (z * u16::MAX as f32) as u16;
                Rgba([z, z, z, u16::MAX])
            }
        };
        // safety: `image` has the same size as all the others
        unsafe { image.unsafe_put_pixel(x, y, pixel) };
    }

    image
}

#[must_use]
pub struct ParallelRenderer {
    threads: Vec<JoinHandle<()>>,
    render_receiver: mpsc::Receiver<Arc<Mutex<Runtime>>>,
    job_sender: watch::WatchSender<Option<(Config, Arc<AtomicUsize>)>>,
}
impl ParallelRenderer {
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
                    let (config, job_counter): (Config, Arc<AtomicUsize>) =
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
    fn send(&mut self, job: Config, job_counter: Arc<AtomicUsize>) {
        self.job_sender.send(Some((job, job_counter)));
    }
    fn recv(&mut self) -> impl Iterator<Item = Arc<Mutex<Runtime>>> + '_ {
        self.render_receiver.iter().take(self.num_threads())
    }
    fn num_threads(&self) -> usize {
        self.threads.len()
    }
    pub fn shutdown(self) {
        self.job_sender.send(None);
        self.threads
            .into_iter()
            .for_each(|thread| thread.join().expect("render thread panicked"));
    }
}
impl Default for ParallelRenderer {
    fn default() -> Self {
        Self::new()
    }
}
/// I recommend `16` for `jobs_per_thread`. If you get uneven images with low iteration counts, try
/// `8`.
///
/// At relatively low rations between iterations and pixels (<50), this isn't much faster.
#[allow(clippy::missing_panics_doc)] // it won't panic
#[must_use]
pub fn render_parallel(
    renderer: &mut ParallelRenderer,
    mut config: Config,
    jobs_per_thread: usize,
) -> FinalImage {
    let iterations = config.iterations;
    config.iterations = iterations / renderer.num_threads() / jobs_per_thread;

    // We keep a job counter to balance threads. If one is way slower, it'll "take" less jobs,
    // which results in better runtime.
    let job_counter = Arc::new(AtomicUsize::new(jobs_per_thread * renderer.num_threads()));

    renderer.send(config.clone(), job_counter);

    let mut iter = renderer.recv();
    // UNWRAP: `available_parallelism` is guaranteed to always return >0
    let current = iter.next().unwrap();
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
