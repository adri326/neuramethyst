use nalgebra::DVector;
use rust_mnist::Mnist;

use neuramethyst::{
    argmax, cycle_shuffling,
    derivable::{
        activation::{Linear, Logistic, Relu, Swish, Tanh},
        loss::{CrossEntropy, Euclidean},
    },
    plot_losses,
    prelude::*,
};

const TRAIN_SIZE: usize = 50000;
const TEST_SIZE: usize = 1000;
const WIDTH: usize = 28;
const HEIGHT: usize = 28;
const LATENT_SIZE: usize = 25;

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

    let test_data = test_images
        .clone()
        .zip(test_images.clone())
        .collect::<Vec<_>>();

    // First, train an encoder-decoder network (unsupervised)

    let mut network = neura_sequential![
        neura_layer!("dense", 100).activation(Swish(Logistic)),
        neura_layer!("dense", 50).activation(Swish(Logistic)),
        neura_layer!("dense", LATENT_SIZE).activation(Tanh),
        neura_layer!("dense", 50),
        neura_layer!("dense", 100),
        neura_layer!("dense", WIDTH * HEIGHT).activation(Relu),
    ]
    .construct(NeuraShape::Vector(WIDTH * HEIGHT))
    .unwrap();

    let trainer = NeuraBatchedTrainer::with_epochs(0.03, 75, 512, TRAIN_SIZE);
    // trainer.log_iterations = 1;

    let losses = trainer.train(
        &NeuraBackprop::new(Euclidean),
        &mut network,
        cycle_shuffling(
            train_images.clone().zip(train_images.clone()),
            rand::thread_rng(),
        ),
        &test_data,
    );

    plot_losses(losses, 128, 48);

    // Then, train a small network to decode the encoded data into the categories

    let trimmed_network = network.clone().trim_tail().trim_tail().trim_tail();

    let mut network = neura_sequential![
        ..trimmed_network.lock(),
        neura_layer!("dense", LATENT_SIZE)
            .activation(Tanh)
            .construct(NeuraShape::Vector(LATENT_SIZE))
            .unwrap(),
        neura_layer!("dense", 10)
            .activation(Linear)
            .construct(NeuraShape::Vector(LATENT_SIZE))
            .unwrap(),
        neura_layer!("softmax")
    ];
    let test_data = test_images
        .clone()
        .zip(test_labels.clone())
        .collect::<Vec<_>>();

    let trainer = NeuraBatchedTrainer::with_epochs(0.03, 20, 128, TRAIN_SIZE);

    plot_losses(
        trainer.train(
            &NeuraBackprop::new(Euclidean),
            &mut network,
            cycle_shuffling(train_images.clone().zip(train_labels), rand::thread_rng()),
            &test_data,
        ),
        128,
        48,
    );

    let mut correct = 0;
    for (test_image, test_label) in test_images.zip(test_labels) {
        let guess = network.eval(&test_image);
        let guess = argmax(guess.as_slice());
        let actual = argmax(test_label.as_slice());

        if guess == actual {
            correct += 1;
        }
    }

    println!("");
    println!(
        "{} correct out of {}: {:.2}%",
        correct,
        TEST_SIZE,
        (correct as f32 / TEST_SIZE as f32) * 100.0
    );
}

fn one_hot(value: usize, categories: usize) -> DVector<f32> {
    let mut res = DVector::from_element(categories, 0.0);
    if value < categories {
        res[value] = 1.0;
    }
    res
}
