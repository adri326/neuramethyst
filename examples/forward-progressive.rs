use nalgebra::{dvector, DVector};
use neuramethyst::derivable::activation::{LeakyRelu, Logistic, Swish, Tanh};
use neuramethyst::derivable::regularize::*;
use neuramethyst::gradient_solver::NeuraForwardForward;
use neuramethyst::prelude::*;
use rand::Rng;

const EPOCHS: usize = 10;
const REG_FACTOR: f32 = 0.003;

macro_rules! iteration {
    ( $network:ident, $width:expr, $trainer:expr, $gradient_solver:expr, $test_inputs:expr ) => {
        let mut $network = neura_sequential![
            ..($network.lock()),
            neura_layer!("normalize")
                .construct(NeuraShape::Vector($width))
                .unwrap(),
            neura_layer!("dense", $width)
                .activation(Swish(Logistic))
                .regularization(NeuraL2(REG_FACTOR))
                .construct(NeuraShape::Vector($width))
                .unwrap()
        ];
        for _epoch in 0..EPOCHS {
            $trainer.train(&$gradient_solver, &mut $network, generator(), &$test_inputs);

            draw_network(&$network);
        }
    };
}

pub fn main() {
    let width: usize = 60;

    let mut network = neura_sequential![neura_layer!("dense", width).activation(LeakyRelu(0.1)),]
        .construct(NeuraShape::Vector(2))
        .unwrap();

    let test_inputs = generator().filter(|x| x.1).take(50).collect::<Vec<_>>();

    let gradient_solver = NeuraForwardForward::new(Tanh, 0.5);
    let mut trainer = NeuraBatchedTrainer::new().learning_rate(0.01).iterations(0);
    trainer.batch_size = 256;

    for _epoch in 0..EPOCHS {
        trainer.train(&gradient_solver, &mut network, generator(), &test_inputs);

        draw_network(&network);
    }

    iteration!(network, width, trainer, gradient_solver, test_inputs);
    iteration!(network, width, trainer, gradient_solver, test_inputs);
    iteration!(network, width, trainer, gradient_solver, test_inputs);
    iteration!(network, width, trainer, gradient_solver, test_inputs);
    // iteration!(network, width, trainer, gradient_solver, test_inputs);
    // iteration!(network, width, trainer, gradient_solver, test_inputs);
}

fn generator() -> impl Iterator<Item = (DVector<f32>, bool)> {
    let mut rng = rand::thread_rng();
    std::iter::repeat_with(move || {
        let good = rng.gen_bool(0.5);
        // Clifford attractor
        let (a, b, c, d) = (1.5, -1.8, 1.6, 0.9);

        let noise = 0.0005;
        let mut x: f32 = rng.gen_range(-noise..noise);
        let mut y: f32 = rng.gen_range(-noise..noise);
        for _ in 0..rng.gen_range(150..200) {
            let nx = (a * y).sin() + c * (a * x).cos();
            let ny = (b * x).sin() + d * (b * y).cos();
            x = nx;
            y = ny;
        }

        // Bad samples are shifted by a random amount
        if !good {
            let radius = rng.gen_range(0.4..0.5);
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            x += angle.cos() * radius;
            y += angle.sin() * radius;
        }

        (dvector![x, y], good)
    })
}

// TODO: move this to the library?
fn draw_neuron_activation<F: Fn([f64; 2]) -> Vec<f64>>(callback: F, scale: f64) {
    use viuer::Config;

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;

    let mut image = image::RgbImage::new(WIDTH, HEIGHT);

    fn sigmoid(x: f64) -> f64 {
        0.1 + 0.9 * x.abs().powf(0.8)
    }

    for y in 0..HEIGHT {
        let y2 = 2.0 * y as f64 / HEIGHT as f64 - 1.0;
        for x in 0..WIDTH {
            let x2 = 2.0 * x as f64 / WIDTH as f64 - 1.0;
            let activation = callback([x2 * scale, y2 * scale]);
            let r = (sigmoid(activation.get(0).copied().unwrap_or(-1.0)) * 255.0).floor() as u8;
            let g = (sigmoid(activation.get(1).copied().unwrap_or(-1.0)) * 255.0).floor() as u8;
            let b = (sigmoid(activation.get(2).copied().unwrap_or(-1.0)) * 255.0).floor() as u8;

            *image.get_pixel_mut(x, y) = image::Rgb([r, g, b]);
        }
    }

    let config = Config {
        use_kitty: false,
        truecolor: true,
        // absolute_offset: false,
        ..Default::default()
    };

    viuer::print(&image::DynamicImage::ImageRgb8(image), &config).unwrap();
}

fn draw_network<Network: NeuraLayer<DVector<f32>, Output = DVector<f32>>>(network: &Network) {
    draw_neuron_activation(
        |input| {
            let result = network.eval(&dvector![input[0] as f32, input[1] as f32]);
            let result_good = result.map(|x| x * x).sum();

            let result_norm = result / result_good.sqrt();
            let mut result_rgb = DVector::from_element(3, 0.0);

            for i in 0..result_norm.len() {
                result_rgb[i % 3] += result_norm[i].abs();
            }

            (result_rgb * result_good.tanh() * 12.0 / result_norm.len() as f32)
                .into_iter()
                .map(|x| *x as f64)
                .collect()
        },
        2.0,
    );
}
