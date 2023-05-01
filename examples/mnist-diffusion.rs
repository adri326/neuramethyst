use nalgebra::DVector;
use rand::Rng;
use rust_mnist::Mnist;
use std::io::Write;

use neuramethyst::{
    cycle_shuffling,
    derivable::{
        activation::{Logistic, Relu, Swish},
        loss::Euclidean,
        regularize::NeuraL2,
    },
    plot_losses,
    prelude::*,
};

const TRAIN_SIZE: usize = 50000;
const TEST_SIZE: usize = 1000;
const WIDTH: usize = 28;
const HEIGHT: usize = 28;
const REG_RATE: f32 = 0.003;
const EPOCHS: usize = 80;

// const BASE_NOISE: f32 = 0.05;
const NOISE_AMOUNT: f32 = 0.5;
const SHIFT_AMOUNT: i32 = 9;

pub fn main() {
    let Mnist {
        train_data: train_images,
        train_labels,
        test_data: test_images,
        test_labels,
        ..
    } = Mnist::new("data/");

    let train_images = train_images
        .into_iter()
        .map(|raw| {
            DVector::from_iterator(WIDTH * HEIGHT, raw.into_iter().map(|x| x as f32 / 255.0))
        })
        .take(TRAIN_SIZE);
    let train_labels = train_labels
        .into_iter()
        .map(|x| one_hot(x as usize, 10))
        .take(TRAIN_SIZE);

    let test_images = test_images
        .into_iter()
        .map(|raw| {
            DVector::from_iterator(WIDTH * HEIGHT, raw.into_iter().map(|x| x as f32 / 255.0))
        })
        .take(TEST_SIZE);
    let test_labels = test_labels
        .into_iter()
        .map(|x| one_hot(x as usize, 10))
        .take(TEST_SIZE);

    let test_data: Vec<_> = augment_data(test_images.zip(test_labels)).collect();

    let mut network = neura_residual![
        <= 0, 1;
        neura_layer!("isolate", WIDTH * HEIGHT, WIDTH * HEIGHT + 10) => 1, 3, 5, 7, 9, 10;
        neura_layer!("isolate", 0, WIDTH * HEIGHT) => 0, 1, 3;
        neura_layer!("dense", 100).regularization(NeuraL2(REG_RATE)).activation(Swish(Logistic)) => 0, 2;
        neura_layer!("dropout", 0.5);
        neura_layer!("dense", 50).regularization(NeuraL2(REG_RATE)).activation(Swish(Logistic)) => 0, 2, 4;
        neura_layer!("dropout", 0.5);
        neura_layer!("dense", 50).regularization(NeuraL2(REG_RATE)).activation(Swish(Logistic)) => 0, 2;
        neura_layer!("dropout", 0.33);
        neura_layer!("dense", 25).regularization(NeuraL2(REG_RATE)).activation(Swish(Logistic)) => 0, 2;
        neura_layer!("dropout", 0.33);
        neura_layer!("dense", 25).regularization(NeuraL2(REG_RATE)).activation(Swish(Logistic));
        // neura_layer!("dropout", 0.33);
        neura_layer!("dense", WIDTH * HEIGHT).activation(Relu);
    ]
    .construct(NeuraShape::Vector(WIDTH * HEIGHT + 10))
    .unwrap();

    let trainer = NeuraBatchedTrainer::with_epochs(0.03, EPOCHS, 512, TRAIN_SIZE);
    // trainer.log_iterations = 1;
    let train_data = augment_data(cycle_shuffling(
        train_images.clone().zip(train_labels.clone()),
        rand::thread_rng(),
    ));

    let losses = trainer.train(
        &NeuraBackprop::new(Euclidean),
        &mut network,
        train_data,
        &test_data,
    );
    plot_losses(losses, 128, 48);

    loop {
        let mut image = uniform_vector(WIDTH * HEIGHT + 10);
        let mut buffer = String::new();
        print!("> ");
        std::io::stdout().flush().unwrap();
        if let Err(_) = std::io::stdin().read_line(&mut buffer) {
            break;
        }

        for i in 0..10 {
            image[WIDTH * HEIGHT + i] = buffer
                .chars()
                .any(|c| c == char::from_digit(i as u32, 10).unwrap())
                as u8 as f32;
        }

        for _iter in 0..5 {
            let new_image = network.eval(&image);

            neuramethyst::draw_neuron_activation(
                |[x, y]| {
                    let x = ((x + 1.0) / 2.0 * WIDTH as f32) as usize;
                    let y = ((y + 1.0) / 2.0 * HEIGHT as f32) as usize;

                    let index = x + y * WIDTH;

                    vec![new_image[index]]
                },
                1.0,
                WIDTH as u32,
                HEIGHT as u32,
            );

            for i in 0..(WIDTH * HEIGHT) {
                image[i] = new_image[i] * 0.6 + image[i] * 0.3;
            }

            std::thread::sleep(std::time::Duration::new(0, 100_000_000));
        }
    }
}

fn uniform_vector(length: usize) -> DVector<f32> {
    let mut res = DVector::from_element(length, 0.0);
    let mut rng = rand::thread_rng();

    for i in 0..length {
        res[i] = rng.gen();
    }

    res
}

fn one_hot(value: usize, categories: usize) -> DVector<f32> {
    let mut res = DVector::from_element(categories, 0.0);
    if value < categories {
        res[value] = 1.0;
    }
    res
}

fn add_noise(mut image: DVector<f32>, rng: &mut impl Rng, amount: f32) -> DVector<f32> {
    if amount <= 0.0 {
        return image;
    }

    let uniform = rand::distributions::Uniform::new(0.0, amount);

    for i in 0..image.len() {
        let x = rng.sample(uniform);
        image[i] = image[i] * (1.0 - x) + (1.0 - image[i]) * x;
    }

    image
}

fn shift(image: &DVector<f32>, dx: i32, dy: i32) -> DVector<f32> {
    let mut res = DVector::from_element(image.len(), 0.0);
    let width = WIDTH as i32;
    let height = HEIGHT as i32;

    for y in 0..height {
        for x in 0..width {
            let x2 = x + dx;
            let y2 = y + dy;
            if y2 < 0 || y2 >= height || x2 < 0 || x2 >= width {
                continue;
            }
            res[(y2 * width + x2) as usize] = image[(y * width + x) as usize];
        }
    }

    res
}

fn augment_data(
    iter: impl Iterator<Item = (DVector<f32>, DVector<f32>)>,
) -> impl Iterator<Item = (DVector<f32>, DVector<f32>)> {
    let mut rng = rand::thread_rng();
    iter.map(move |(image, label)| {
        let noise_amount = rng.gen_range(0.05..NOISE_AMOUNT);
        let base_image = shift(
            &image,
            rng.gen_range(-SHIFT_AMOUNT..SHIFT_AMOUNT),
            rng.gen_range(-SHIFT_AMOUNT..SHIFT_AMOUNT),
        ) * rng.gen_range(0.6..1.0);
        // let base_image = add_noise(base_image, &mut rng, base_noise);

        let noisy_image = add_noise(base_image.clone(), &mut rng, noise_amount);

        (
            DVector::from_iterator(
                WIDTH * HEIGHT + 10,
                noisy_image.iter().copied().chain(label.iter().copied()),
            ),
            image,
        )
    })
}
