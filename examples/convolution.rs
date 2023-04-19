#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]

use neuramethyst::algebra::NeuraVector;
use neuramethyst::derivable::reduce::{Average, Max};
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
        neura_layer!("unstable_reshape", 1, { 28 * 28 }),
        neura_layer!("conv2d_pad", 1, {28 * 28}; 28, 3; neura_layer!("dense", {1 * 3 * 3}, 3; Relu)),
        // neura_layer!("conv2d_block", 7, 7; 4; neura_layer!("dense", {3 * 4 * 4}, 8; Relu)),
        // neura_layer!("conv2d_pad"; 28, 1; neura_layer!("dense", {30 * 1 * 1}, 10; Relu)),
        neura_layer!("unstable_flatten"),
        neura_layer!("dropout", 0.33),
        neura_layer!("unstable_reshape", 3, { 28 * 28 }),
        neura_layer!("conv2d_block", 14, 14; 2; neura_layer!("dense", {3 * 2 * 2}, 2; Relu)),
        // neura_layer!("unstable_flatten"),
        // neura_layer!("dropout", 0.33),
        // neura_layer!("unstable_reshape", 2, { 14 * 14 }),
        // neura_layer!("conv2d_pad"; 14, 5; neura_layer!("dense", {2 * 5 * 5}, 20; Relu)),
        // neura_layer!("pool_global"; Max),

        // neura_layer!("pool1d", {14 * 2}, 7; Max),
        neura_layer!("unstable_flatten"),
        neura_layer!("dropout", 0.2),
        neura_layer!("dense", 10).activation(Linear),
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
