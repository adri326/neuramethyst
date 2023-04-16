#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]

use neuramethyst::algebra::NeuraVector;
use rust_mnist::Mnist;

use neuramethyst::derivable::activation::{Linear, Relu};
use neuramethyst::derivable::loss::CrossEntropy;
use neuramethyst::{cycle_shuffling, one_hot, prelude::*};

fn main() {
    const TRAIN_SIZE: usize = 1000;

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
        neura_layer!("unstable_reshape", 28, 28),
        neura_layer!("conv1d_pad", 3; neura_layer!("dense", {28 * 3}, 10; Relu)),
        neura_layer!("unstable_flatten"),
        // neura_layer!("dense", 100; Relu),
        // neura_layer!("dropout", 0.5),
        neura_layer!("dense", 30; Relu),
        neura_layer!("dropout", 0.5),
        neura_layer!("dense", 10; Linear),
        neura_layer!("softmax")
    ];

    let mut trainer = NeuraBatchedTrainer::with_epochs(0.03, 100, 128, TRAIN_SIZE);
    trainer.learning_momentum = 0.001;

    trainer.train(
        NeuraBackprop::new(CrossEntropy),
        &mut network,
        train_iter,
        &test_inputs,
    );
}
