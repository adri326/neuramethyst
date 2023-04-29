use std::rc::Rc;

use crate::layer::*;

mod wrapper;
pub use wrapper::*;

mod input;
pub use input::*;

mod axis;
pub use axis::*;

mod construct;
pub use construct::NeuraResidualConstructErr;

mod node;
pub use node::*;

mod last;
pub use last::*;

#[macro_export]
macro_rules! neura_residual {
    [ "__combine_layers", ] => {
        $crate::network::residual::NeuraResidualLast::new()
    };

    [ "__combine_layers",
        $layer:expr $(, $axis:expr)? $( => $( $offset:expr ),* )?
        $(; $( $rest_layer:expr $(, $rest_axis:expr)? $( => $( $rest_offset:expr ),* )? );*)?
    ] => {{
        let layer = $crate::network::residual::NeuraResidualNode::new($layer,
            neura_residual![
                "__combine_layers",
                $($( $rest_layer $(, $rest_axis)? $( => $( $rest_offset ),* )? );*)?
            ]
        );

        $(
            let layer = layer.axis($axis);
        )?

        $(
            let layer = layer.offsets(vec![$($offset),*]);
        )?

        layer
    }};

    [
        $( <= $( $initial_offset:expr ),* ;)?
        $( $layer:expr $(, $axis:expr)? $( => $( $offset:expr ),* $(,)? )? );*
        $(;)?
    ] => {{
        let res = $crate::network::residual::NeuraResidual::new(
            neura_residual![ "__combine_layers", $( $layer $(, $axis)? $( => $( $offset ),* )? );* ]
        );

        $(
            let res = res.offsets(vec![$($initial_offset),*]);
        )?

        res
    }};
}

#[cfg(test)]
mod test {
    use nalgebra::dvector;

    use crate::gradient_solver::NeuraGradientSolver;
    use crate::{derivable::loss::Euclidean, neura_layer, prelude::NeuraBackprop};

    use super::*;

    #[test]
    fn test_resnet_eval() {
        let network = NeuraResidual::new(
            NeuraResidualNode::new(
                neura_layer!("dense", 4)
                    .construct(NeuraShape::Vector(2))
                    .unwrap(),
                NeuraResidualNode::new(
                    neura_layer!("dense", 3)
                        .construct(NeuraShape::Vector(4))
                        .unwrap(),
                    NeuraResidualNode::new(
                        neura_layer!("dense", 6)
                            .construct(NeuraShape::Vector(7))
                            .unwrap(),
                        (),
                    ),
                ),
            )
            .offset(1),
        );

        network.eval(&dvector![0.2, 0.4]);
    }

    #[test]
    fn test_resnet_macro() {
        let network = neura_residual![
            <= 0, 2;
            neura_layer!("dense", 5) => 0, 1;
            neura_layer!("dense", 5);
            neura_layer!("dense", 3)
        ];

        println!("{:#?}", network);

        assert_eq!(network.initial_offsets, vec![0, 2]);
        assert_eq!(network.layers.offsets, vec![0, 1]);
        assert_eq!(network.layers.child_network.offsets, vec![0]);
        assert_eq!(
            network.layers.child_network.child_network.child_network,
            NeuraResidualLast::new()
        );

        let network = neura_residual![
            neura_layer!("dense", 4) => 0;
        ];

        assert_eq!(network.initial_offsets, vec![0]);
    }

    #[test]
    fn test_resnet_partial() {
        let network = neura_residual![
            <= 0, 1;
            neura_layer!("dense", 2) => 0, 1;
            neura_layer!("dense", 4);
            neura_layer!("dense", 8)
        ]
        .construct(NeuraShape::Vector(1))
        .unwrap();

        assert_eq!(network.output_shape(), NeuraShape::Vector(8));
        assert_eq!(network.layers.layer.input_len(), 1);
        assert_eq!(network.layers.child_network.layer.input_len(), 3); // input (1) + first layer (2)
        assert_eq!(
            network.layers.child_network.child_network.layer.input_len(),
            6
        ); // first layer (2) + second layer (4)

        assert_eq!(network.layers.input_offsets, vec![0]);
        assert_eq!(network.layers.child_network.input_offsets, vec![1, 0]); // input, first layer
        assert_eq!(
            network.layers.child_network.child_network.input_offsets,
            vec![1, 0]
        ); // first layer, second layer

        let map_shape = |shapes: &[NeuraShape]| {
            shapes
                .into_iter()
                .map(|shape| shape.size())
                .collect::<Vec<_>>()
        };

        assert_eq!(map_shape(&network.layers.input_shapes), vec![1]);
        assert_eq!(
            map_shape(&network.layers.child_network.input_shapes),
            vec![1, 2]
        ); // input, first layer
        assert_eq!(
            map_shape(&network.layers.child_network.child_network.input_shapes),
            vec![2, 4]
        ); // first layer, second layer

        network.eval(&dvector![0.0]);
    }

    #[test]
    fn test_resnet_backprop() {
        let network = neura_residual![
            <= 0, 1;
            neura_layer!("dense", 2) => 0, 1;
            neura_layer!("dense", 4);
            neura_layer!("dense", 4)
        ]
        .construct(NeuraShape::Vector(1))
        .unwrap();

        let backprop = NeuraBackprop::new(Euclidean);

        backprop.get_gradient(&network, &dvector![0.0], &dvector![0.0, 0.0, 0.0, 0.0]);
    }
}
