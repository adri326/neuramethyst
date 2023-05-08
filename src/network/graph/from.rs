use crate::network::residual::{NeuraAxisDefault, NeuraSplitInputs};

use super::*;

pub trait FromSequential<Seq, Data> {
    fn from_sequential_rec(
        seq: &Seq,
        nodes: Vec<NeuraGraphNodeConstructed<Data>>,
        input_shape: NeuraShape,
    ) -> Self;
}

impl<Data> FromSequential<(), Data> for NeuraGraph<Data> {
    fn from_sequential_rec(
        _seq: &(),
        nodes: Vec<NeuraGraphNodeConstructed<Data>>,
        input_shape: NeuraShape,
    ) -> Self {
        Self {
            output_index: nodes.len(),
            buffer_size: nodes.len() + 1,
            nodes: nodes,
            output_shape: input_shape,
        }
    }
}

impl<Data: Clone + 'static, Layer: NeuraLayer<Data, Output = Data>, ChildNetwork>
    FromSequential<NeuraSequential<Layer, ChildNetwork>, Data> for NeuraGraph<Data>
where
    NeuraGraph<Data>: FromSequential<ChildNetwork, Data>,
    NeuraAxisDefault: NeuraSplitInputs<Data, Combined = Data>,
    Layer::IntermediaryRepr: 'static,
{
    fn from_sequential_rec(
        seq: &NeuraSequential<Layer, ChildNetwork>,
        mut nodes: Vec<NeuraGraphNodeConstructed<Data>>,
        input_shape: NeuraShape,
    ) -> Self {
        nodes.push(NeuraGraphNodeConstructed {
            node: Box::new(NeuraGraphNode::from_layer(
                seq.layer.clone(),
                vec![input_shape],
            )),
            inputs: vec![nodes.len()],
            output: nodes.len() + 1,
        });

        Self::from_sequential_rec(&seq.child_network, nodes, seq.layer.output_shape())
    }
}

impl<Data> NeuraGraph<Data> {
    pub fn from_sequential<Layer, ChildNetwork>(
        network: NeuraSequential<Layer, ChildNetwork>,
        input_shape: NeuraShape,
    ) -> Self
    where
        NeuraGraph<Data>: FromSequential<NeuraSequential<Layer, ChildNetwork>, Data>,
        NeuraSequential<Layer, ChildNetwork>: NeuraLayerBase,
    {
        Self::from_sequential_rec(&network, vec![], input_shape)
    }
}
