#![feature(generic_arg_infer)]
// #![feature(generic_const_exprs)]

use neuramethyst::algebra::NeuraVector;
use rust_mnist::Mnist;

use neuramethyst::derivable::activation::{Linear, Relu};
use neuramethyst::derivable::loss::CrossEntropy;
use neuramethyst::{cycle_shuffling, one_hot, prelude::*};

fn main() {
    const TRAIN_SIZE: usize = 100;

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
            raw.into_iter()
                .map(|x| x as f64 / 255.0)
                .collect::<NeuraVector<{ 28 * 28 }, f64>>()
        })
        .take(TRAIN_SIZE);
    let train_labels = train_labels
        .into_iter()
        .map(|x| one_hot::<10>(x as usize))
        .take(TRAIN_SIZE);

    let test_images = test_images
        .into_iter()
        .map(|raw| {
            raw.into_iter()
                .map(|x| x as f64 / 255.0)
                .collect::<NeuraVector<{ 28 * 28 }, f64>>()
        })
        .take(TRAIN_SIZE / 6);
    let test_labels = test_labels
        .into_iter()
        .map(|x| one_hot::<10>(x as usize))
        .take(TRAIN_SIZE / 6);

    let train_iter = cycle_shuffling(
        train_images.zip(train_labels.into_iter()),
        rand::thread_rng(),
    );

    let test_inputs: Vec<_> = test_images.zip(test_labels.into_iter()).collect();

    let mut network = neura_sequential![
        neura_layer!("dense", { 28 * 28 }, 200; Relu),
        neura_layer!("dropout", 0.5),
        neura_layer!("dense", 100; Relu),
        neura_layer!("dropout", 0.5),
        neura_layer!("dense", 30; Relu),
        neura_layer!("dropout", 0.5),
        neura_layer!("dense", 10; Linear),
        neura_layer!("softmax")
    ];

    let mut trainer = NeuraBatchedTrainer::new(0.03, TRAIN_SIZE * 10);
    trainer.log_iterations = (TRAIN_SIZE / 128).max(1);
    trainer.batch_size = 128;
    trainer.learning_momentum = 0.001;

    trainer.train(
        NeuraBackprop::new(CrossEntropy),
        &mut network,
        train_iter,
        &test_inputs,
    );
}
