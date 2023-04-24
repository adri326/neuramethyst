use std::rc::Rc;

use nalgebra::{DVector, Scalar};
use num::Float;

use crate::layer::*;

mod layer_impl;

mod input;
pub use input::*;

mod axis;
pub use axis::*;

mod construct;
pub use construct::NeuraResidualConstructErr;

#[derive(Clone, Debug, PartialEq)]
pub struct NeuraResidual<Layers> {
    /// Instance of NeuraResidualNode
    layers: Layers,

    /// Array of which layers to send the input to, defaults to `vec![0]`
    initial_offsets: Vec<usize>,
}

impl<Layers> NeuraResidual<Layers> {
    pub fn new(layers: Layers) -> Self {
        Self {
            layers,
            initial_offsets: vec![0],
        }
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.initial_offsets.push(offset);
        self
    }

    pub fn offsets(mut self, offsets: Vec<usize>) -> Self {
        self.initial_offsets = offsets;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NeuraResidualNode<Layer, ChildNetwork, Axis> {
    pub layer: Layer,
    pub child_network: ChildNetwork,

    /// Array of relative layers indices to send the offset of this layer to,
    /// defaults to `vec![0]`.
    offsets: Vec<usize>,

    pub axis: Axis,

    output_shape: Option<NeuraShape>,
}

impl<Layer, ChildNetwork> NeuraResidualNode<Layer, ChildNetwork, NeuraAxisAppend> {
    pub fn new(layer: Layer, child_network: ChildNetwork) -> Self {
        Self {
            layer,
            child_network,
            offsets: vec![0],
            axis: NeuraAxisAppend,
            output_shape: None,
        }
    }
}

impl<Layer, ChildNetwork, Axis> NeuraResidualNode<Layer, ChildNetwork, Axis> {
    pub fn offsets(mut self, offsets: Vec<usize>) -> Self {
        self.offsets = offsets;
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offsets.push(offset);
        self
    }

    pub fn axis<Axis2>(self, axis: Axis2) -> NeuraResidualNode<Layer, ChildNetwork, Axis2> {
        NeuraResidualNode {
            layer: self.layer,
            child_network: self.child_network,
            offsets: self.offsets,
            axis,
            // Drop the knowledge of output_shape
            output_shape: None,
        }
    }
}

#[macro_export]
macro_rules! neura_residual {
    [ "__combine_layers", ] => {
        ()
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

    use crate::neura_layer;

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
        assert_eq!(network.layers.child_network.child_network.child_network, ());

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

        network.eval(&dvector![0.0]);
    }
}
