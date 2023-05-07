use crate::err::NeuraGraphErr;
use std::collections::{HashMap, HashSet, VecDeque};

use super::*;

pub struct NeuraGraphPartial<Data> {
    pub nodes: Vec<Box<dyn NeuraGraphNodePartial<Data>>>,
    pub output: String,
    pub input: String,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub(crate) enum GraphIndex {
    Input,
    Node(usize),
}

impl<Data> NeuraGraphPartial<Data> {
    pub(crate) fn get_index_map(&self) -> Result<HashMap<String, GraphIndex>, NeuraGraphErr> {
        let mut result = HashMap::with_capacity(self.nodes.len());

        result.insert(self.input.clone(), GraphIndex::Input);

        for (index, node) in self.nodes.iter().enumerate() {
            if result.contains_key(node.name()) {
                return Err(NeuraGraphErr::InvalidName(node.name().to_string()));
            }
            result.insert(node.name().to_string(), GraphIndex::Node(index));
        }

        Ok(result)
    }

    pub(crate) fn get_reverse_graph(
        &self,
        index_map: &HashMap<String, GraphIndex>,
    ) -> Result<HashMap<GraphIndex, HashSet<GraphIndex>>, NeuraGraphErr> {
        let mut result = HashMap::new();

        result.insert(GraphIndex::Input, HashSet::new());

        for i in 0..self.nodes.len() {
            result.insert(GraphIndex::Node(i), HashSet::new());
        }

        for (index, node) in self.nodes.iter().enumerate() {
            for input in node.inputs() {
                let input_index = index_map
                    .get(input)
                    .copied()
                    .ok_or_else(|| NeuraGraphErr::MissingNode(input.clone()))?;
                result
                    .get_mut(&input_index)
                    .expect("index_map returned invalid values")
                    .insert(GraphIndex::Node(index));
            }
        }

        Ok(result)
    }

    pub(crate) fn get_node_order(
        &self,
        index_map: &HashMap<String, GraphIndex>,
        reverse_graph: &HashMap<GraphIndex, HashSet<GraphIndex>>,
    ) -> Result<Vec<usize>, NeuraGraphErr> {
        let mut result: Vec<usize> = Vec::new();
        let mut closed: HashSet<GraphIndex> = HashSet::with_capacity(self.nodes.len());
        let mut open = VecDeque::with_capacity(self.nodes.len());
        open.push_front(GraphIndex::Input);

        /*
        index_map.get(&self.output)
            .copied()
            .ok_or_else(|| NeuraGraphErr::MissingNode(self.output.clone()))?
        */

        while let Some(current) = open.pop_back() {
            if closed.contains(&current) {
                continue;
            }

            closed.insert(current);
            // Do not put 0 (the input) in result
            if let GraphIndex::Node(index) = current {
                result.push(index);
            }

            println!("{:?}", current);

            for next_node in reverse_graph[&current].iter().copied() {
                // Ignore nodes that are already in the closed set
                if closed.contains(&next_node) {
                    continue;
                }

                let GraphIndex::Node(node_index) = next_node else {
                    panic!("Unreachable: cannot have GraphIndex::Input as the output of a node");
                };

                let inputs = self.nodes[node_index].inputs();

                // Only consider nodes whose inputs are in the closed set (meaning they would be ready to be evaluated)
                if !inputs
                    .iter()
                    .all(|input| closed.contains(&index_map[input]))
                {
                    continue;
                }

                open.push_front(next_node);
            }
        }

        if result.len() != self.nodes.len() {
            // TODO: verify that if result.len() != self.nodes.len(), then there is a cyclic subgraph

            return Err(NeuraGraphErr::Cyclic);
        }

        Ok(result)
    }
}

impl<Data> NeuraPartialLayer for NeuraGraphPartial<Data> {
    type Constructed = NeuraGraph<Data>;

    type Err = NeuraGraphErr;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        let index_map = self.get_index_map()?;
        let reverse_graph = self.get_reverse_graph(&index_map)?;

        // List out the nodes in their execution order
        let node_order = self.get_node_order(&index_map, &reverse_graph)?;
        let mut new_index_map: HashMap<String, usize> = HashMap::from_iter(
            node_order
                .iter()
                .map(|&i| (self.nodes[i].name().to_string(), i)),
        );
        new_index_map.insert(self.input.clone(), 0);

        // TODO: filter out the nodes that are not necessary for computing the result (BFS from the output node back to the inputs)
        // A temporary solution can be to trim the graph
        let output_index = new_index_map
            .get(&self.output)
            .copied()
            .ok_or_else(|| NeuraGraphErr::MissingNode(self.output.clone()))?;

        let mut nodes = Vec::with_capacity(self.nodes.len());
        let mut shapes: Vec<Option<NeuraShape>> = vec![None; self.nodes.len() + 1];
        shapes[0] = Some(input_shape);

        for index in node_order.into_iter() {
            let node = &*self.nodes[index];
            let node_inputs = node.inputs();
            let mut inputs = Vec::with_capacity(node_inputs.len());
            let mut input_shapes = Vec::with_capacity(node_inputs.len());

            for input in node_inputs {
                let input_index = new_index_map.get(input).copied().expect(
                    "Unreachable: new_index_map should contain all nodes defined and all nodes should have existing nodes as input"
                );
                inputs.push(input_index);
                input_shapes.push(shapes[input_index].expect(
                    "Unreachable: the order of execution should guarantee that all inputs have appeared before")
                );
            }

            let (constructed, output_shape) = node
                .construct(input_shapes)
                .map_err(|e| NeuraGraphErr::LayerErr(e))?;

            shapes[index] = Some(output_shape);

            nodes.push(NeuraGraphNodeConstructed {
                node: constructed,
                inputs,
                output: new_index_map
                    .get(node.name())
                    .copied()
                    .unwrap_or_else(|| unreachable!()),
            });
        }

        let output_shape = shapes[output_index].unwrap_or_else(|| unreachable!());

        Ok(NeuraGraph {
            nodes,
            // input_shape,
            output_shape,
            output_index,
            buffer_size: self.nodes.len() + 1,
        })
    }
}
