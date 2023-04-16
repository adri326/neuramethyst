#![feature(generic_arg_infer)]

use neuramethyst::algebra::NeuraVector;
use neuramethyst::derivable::activation::Relu;
use neuramethyst::derivable::loss::Euclidean;
use neuramethyst::{cycle_shuffling, prelude::*};

fn main() {
    let mut network = neura_sequential![
        neura_layer!("dense", 2, 4; Relu),
        neura_layer!("dense", 3; Relu),
        neura_layer!("dense", 1; Relu)
    ];

    let inputs: [(NeuraVector<2, f64>, NeuraVector<1, f64>); 4] = [
        ([0.0, 0.0].into(), [0.0].into()),
        ([0.0, 1.0].into(), [1.0].into()),
        ([1.0, 0.0].into(), [1.0].into()),
        ([1.0, 1.0].into(), [0.0].into()),
    ];

    for (input, target) in &inputs {
        println!(
            "Input: {:?}, target: {}, actual: {:.3}",
            &input,
            target[0],
            network.eval(&input)[0]
        );
    }

    let mut trainer = NeuraBatchedTrainer::new(0.05, 1000);
    trainer.batch_size = 6;
    trainer.log_iterations = 250;
    trainer.learning_momentum = 0.01;

    trainer.train(
        NeuraBackprop::new(Euclidean),
        &mut network,
        cycle_shuffling(inputs.iter().cloned(), rand::thread_rng()),
        &inputs,
    );

    for (input, target) in inputs {
        println!(
            "Input: {:?}, target: {}, actual: {:.3}",
            &input,
            target[0],
            network.eval(&input)[0]
        );
    }
}
