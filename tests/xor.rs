use std::fs::File;

use approx::assert_relative_eq;
use nalgebra::{dvector, DMatrix, DVector};
use neuramethyst::{
    derivable::{
        activation::{Relu, Tanh},
        loss::Euclidean,
        regularize::NeuraL0,
    },
    layer::dense::NeuraDenseLayer,
    prelude::*,
};

fn load_test_data() -> Vec<(DMatrix<f64>, DVector<f64>, DMatrix<f64>, DVector<f64>)> {
    let file = File::open("tests/xor.json").unwrap();
    let data: Vec<(DMatrix<f64>, DVector<f64>, DMatrix<f64>, DVector<f64>)> =
        serde_json::from_reader(&file).unwrap();

    data
}

#[test]
fn test_xor_training() {
    let data = load_test_data();

    let mut network = neura_sequential![
        NeuraDenseLayer::new(data[0].0.clone(), data[0].1.clone(), Relu, NeuraL0),
        NeuraDenseLayer::new(data[0].2.clone(), data[0].3.clone(), Tanh, NeuraL0),
    ];

    let inputs = [
        (dvector![0.0, 0.0], dvector![0.0]),
        (dvector![0.0, 1.0], dvector![1.0]),
        (dvector![1.0, 0.0], dvector![1.0]),
        (dvector![1.0, 1.0], dvector![0.0]),
    ];

    let mut trainer = NeuraBatchedTrainer::new(0.05, 1);
    trainer.batch_size = 1;

    for iteration in 0..4 {
        trainer.train(
            &NeuraBackprop::new(Euclidean),
            &mut network,
            inputs.iter().cloned().skip(iteration).take(1),
            &inputs,
        );

        let expected = data[iteration + 1].clone();
        let actual = (
            network.layer.weights.clone(),
            network.layer.bias.clone(),
            network.child_network.layer.weights.clone(),
            network.child_network.layer.bias.clone(),
        );

        assert_relative_eq!(expected.0.as_slice(), actual.0.as_slice());
        assert_relative_eq!(expected.1.as_slice(), actual.1.as_slice());
        assert_relative_eq!(expected.2.as_slice(), actual.2.as_slice());
        assert_relative_eq!(expected.3.as_slice(), actual.3.as_slice());
    }
}
