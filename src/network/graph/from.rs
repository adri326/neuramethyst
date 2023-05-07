use crate::network::residual::{NeuraAxisDefault, NeuraSplitInputs};

use super::*;

trait FromSequential<Seq, Data> {
    fn from_sequential(
        seq: &Seq,
        nodes: Vec<NeuraGraphNodeConstructed<Data>>,
        output_shape: NeuraShape,
    ) -> Self;
}

impl<Data> FromSequential<(), Data> for NeuraGraph<Data> {
    fn from_sequential(
        _seq: &(),
        nodes: Vec<NeuraGraphNodeConstructed<Data>>,
        output_shape: NeuraShape,
    ) -> Self {
        Self {
            output_index: nodes.len(),
            buffer_size: nodes.len() + 1,
            nodes: nodes,
            output_shape,
        }
    }
}

impl<
        Data: Clone,
        Layer: NeuraLayer<Data, Output = Data> + Clone + std::fmt::Debug + 'static,
        ChildNetwork,
    > FromSequential<NeuraSequential<Layer, ChildNetwork>, Data> for NeuraGraph<Data>
where
    NeuraGraph<Data>: FromSequential<ChildNetwork, Data>,
    NeuraAxisDefault: NeuraSplitInputs<Data, Combined = Data>,
{
    fn from_sequential(
        seq: &NeuraSequential<Layer, ChildNetwork>,
        mut nodes: Vec<NeuraGraphNodeConstructed<Data>>,
        output_shape: NeuraShape,
    ) -> Self {
        nodes.push(NeuraGraphNodeConstructed {
            node: Box::new(NeuraGraphNode::from(seq.layer.clone())),
            inputs: vec![nodes.len()],
            output: nodes.len() + 1,
        });

        Self::from_sequential(&seq.child_network, nodes, output_shape)
    }
}

impl<Data, Layer, ChildNetwork> From<NeuraSequential<Layer, ChildNetwork>> for NeuraGraph<Data>
where
    NeuraGraph<Data>: FromSequential<NeuraSequential<Layer, ChildNetwork>, Data>,
    NeuraSequential<Layer, ChildNetwork>: NeuraShapedLayer,
{
    fn from(network: NeuraSequential<Layer, ChildNetwork>) -> Self {
        let output_shape = network.output_shape();
        Self::from_sequential(&network, vec![], output_shape)
    }
}
