#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]

use std::f64::consts::PI;
use std::mem;
use std::ops::Index;

use image::{ImageBuffer, Luma, Pixel, Rgba};
use rand::{Rng, SeedableRng};

pub trait F64Ext {
    #[must_use]
    fn square(self) -> Self;
}
impl F64Ext for f64 {
    fn square(self) -> Self {
        self * self
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
impl Vec3 {
    /// Length of this vector.
    #[must_use]
    pub fn magnitude(self) -> f64 {
        (self.x.square() + self.y.square() + self.z.square()).sqrt()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Matrix3x3 {
    columns: [[f64; 3]; 3],
}
impl Matrix3x3 {
    #[must_use]
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
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.columns[index.0][index.1]
    }
}

pub struct Config {
    pub iterations: usize,
    pub width: u32,
    pub height: u32,

    pub coefficients: Coeffients,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            iterations: 100_000_000,
            width: 1920,
            height: 1080,

            coefficients: Coeffients::default(),
        }
    }
}
#[must_use]
pub fn default_color_transform(delta: Vec3, coeffs: &Coeffients) -> f64 {
    fn part(p: Vec3, coeffs: &Coeffients) -> f64 {
        #[allow(unused)] // clarity
        const RADIAN_45_5: f64 = 91. * PI / 360.;
        /// [`RADIAN_45_5`].cos()
        ///
        /// This was calculated using Rust on x64 Linux.
        #[allow(clippy::excessive_precision)]
        const COS: f64 = 0.700_909_264_299_850_898_183_308_345_323_894_172_906_875_610_351_562_5;
        /// [`RADIAN_45_5`].sin()
        ///
        /// This was calculated using Rust on x64 Linux.
        #[allow(clippy::excessive_precision)]
        const SIN: f64 = 0.713_250_449_154_181_564_992_427_411_198_150_366_544_723_510_742_187_5;

        let x2 = (p.x + coeffs.center_camera.0) * COS + (p.z + coeffs.center_camera.1) * SIN;
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
    (part(delta, coeffs) + delta.magnitude()) / 2.
}
pub struct CoeffientList {
    pub list: [f64; 10],
}
pub struct Coeffients {
    pub x: CoeffientList,
    pub y: CoeffientList,
    pub z: CoeffientList,

    /// (x, y) on where to center the camera
    ///
    /// Is highly related to which coefficients are chosen
    pub center_camera: (f64, f64),

    pub transform_colors: fn(Vec3, &Self) -> f64,
}
impl Default for Coeffients {
    fn default() -> Self {
        Self {
            x: CoeffientList {
                list: [
                    0.021, 1.182, -1.183, 0.128, -1.12, -0.641, -1.152, -0.834, -0.97, 0.722,
                ],
            },
            y: CoeffientList {
                list: [
                    0.243_038, -0.825, -1.2, -0.835_443, -0.835_443, -0.364_557, 0.458, 0.622_785,
                    -0.394_937, -1.032_911,
                ],
            },
            z: CoeffientList {
                list: [
                    -0.455_696, 0.673, 0.915, -0.258_228, -0.495, -0.264, -0.432, -0.416, -0.877,
                    -0.3,
                ],
            },

            center_camera: (-0.005, 0.262),
            transform_colors: default_color_transform,
        }
    }
}

/// This is part of the general algorithm. We get the new coordinates by multiplying previous
/// (using polynomials) with a set of coefficients.
#[must_use]
pub fn next_point(p: Vec3, coefficients: &Coeffients) -> Vec3 {
    fn sum_coefficients(polynomials: &[f64; 10], coefficients: &CoeffientList) -> f64 {
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

/// Can be placed in a [`std::thread_local`].
pub struct Runtime {
    count: ImageBuffer<Luma<u32>, Vec<u32>>,
    steps: ImageBuffer<Luma<f64>, Vec<f64>>,
    zbuf: ImageBuffer<Luma<f64>, Vec<f64>>,

    rng: rand::rngs::SmallRng,
}
impl Runtime {
    pub fn new(config: &Config) -> Self {
        Self {
            count: ImageBuffer::new(config.width, config.height),
            steps: ImageBuffer::new(config.width, config.height),
            zbuf: ImageBuffer::new(config.width, config.height),

            rng: rand::rngs::SmallRng::from_entropy(),
        }
    }
    fn image_identity<T: Pixel>() -> ImageBuffer<T, Vec<T::Subpixel>> {
        ImageBuffer::from_raw(0, 0, Vec::new()).unwrap()
    }
    fn reset(&mut self) {
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

        self.count = ImageBuffer::from_raw(width, height, count).unwrap();
        self.steps = ImageBuffer::from_raw(width, height, steps).unwrap();
        self.zbuf = ImageBuffer::from_raw(width, height, zbuf).unwrap();
    }
}
pub fn render(config: &Config, runtime: &mut Runtime) -> ImageBuffer<Rgba<u16>, Vec<u16>> {
    runtime.reset();

    let mut initial_point = Vec3 {
        x: runtime.rng.gen(),
        y: runtime.rng.gen(),
        z: runtime.rng.gen(),
    };
    for _ in 0..1000 {
        initial_point = next_point(initial_point, &config.coefficients);
    }

    let mut previous_point = initial_point;
    let mut current_point = initial_point;

    ()
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
