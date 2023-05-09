use crate::gradient_solver::NeuraGradientSolver;

use super::*;

#[derive(Debug, Clone)]
pub struct NeuraGraphBackprop<Loss> {
    loss: Loss,
    // TODO: store buffers for re-use, do not clone them
}

impl<Loss> NeuraGraphBackprop<Loss> {
    pub fn new(loss: Loss) -> Self {
        Self { loss }
    }
}

impl<Loss: Clone> From<&NeuraBackprop<Loss>> for NeuraGraphBackprop<Loss> {
    fn from(value: &NeuraBackprop<Loss>) -> Self {
        Self {
            loss: value.get().clone(),
        }
    }
}

impl<
        Data: Clone + std::fmt::Debug + std::ops::Add<Data, Output = Data> + 'static,
        Target,
        Loss: NeuraLoss<Data, Target = Target>,
    > NeuraGradientSolver<Data, Target, NeuraGraph<Data>> for NeuraGraphBackprop<Loss>
{
    // TODO: make it a &mut method
    fn get_gradient(
        &self,
        trainable: &NeuraGraph<Data>,
        input: &Data,
        target: &Target,
    ) -> <NeuraGraph<Data> as NeuraLayerBase>::Gradient {
        let mut output_buffer = trainable.create_buffer();
        let mut backprop_buffer = trainable.create_buffer();
        let mut intermediary_buffer = trainable.create_buffer();
        let mut gradient_buffer = trainable.default_gradient();

        trainable.backprop_in(
            input,
            &self.loss,
            target,
            &mut output_buffer,
            &mut backprop_buffer,
            &mut intermediary_buffer,
            &mut gradient_buffer,
        );

        gradient_buffer
    }

    #[allow(unused)]
    fn score(&self, trainable: &NeuraGraph<Data>, input: &Data, target: &Target) -> f64 {
        todo!()
    }
}

impl<Data> NeuraGraph<Data> {
    fn backprop_in<Loss, Target>(
        &self,
        input: &Data,
        loss: &Loss,
        target: &Target,
        output_buffer: &mut Vec<Option<Data>>,
        backprop_buffer: &mut Vec<Option<Data>>,
        intermediary_buffer: &mut Vec<Option<Box<dyn Any>>>,
        gradient_buffer: &mut Vec<Box<dyn NeuraDynVectorSpace>>,
    ) where
        Data: Clone + std::ops::Add<Data, Output = Data>,
        Loss: NeuraLoss<Data, Target = Target>,
    {
        assert!(output_buffer.len() >= self.nodes.len());
        assert!(backprop_buffer.len() >= self.nodes.len());
        assert!(intermediary_buffer.len() >= self.nodes.len());
        assert!(gradient_buffer.len() >= self.nodes.len());

        output_buffer[0] = Some(input.clone());

        // Forward pass
        for node in self.nodes.iter() {
            // PERF: re-use the allocation for `inputs`, and `.take()` the elements only needed once?
            let inputs: Vec<_> = node
                .inputs
                .iter()
                .map(|&i| {
                    output_buffer[i]
                        .clone()
                        .expect("Unreachable: output of previous layer was not set")
                })
                .collect();
            let (result, intermediary) = node.node.eval_training(&inputs);

            output_buffer[node.output] = Some(result);
            intermediary_buffer[node.output] = Some(intermediary);
        }

        let loss = loss.nabla(
            target,
            output_buffer[self.output_index]
                .as_ref()
                .expect("Unreachable: output was not set"),
        );
        backprop_buffer[self.output_index] = Some(loss);

        // Backward pass
        for node in self.nodes.iter().rev() {
            let Some(epsilon_in) = backprop_buffer[node.output].take() else {
                continue
            };

            // TODO: create more wrapper types to avoid this dereferencing mess
            let intermediary = &**intermediary_buffer[node.output].as_ref().unwrap();

            let epsilon_out = node.node.backprop(intermediary, &epsilon_in);
            let gradient = node.node.get_gradient(intermediary, &epsilon_in);

            (*gradient_buffer[node.output]).add_assign(&*gradient);

            for (&input, epsilon) in node.inputs.iter().zip(epsilon_out.into_iter()) {
                if let Some(existing_gradient) = backprop_buffer[input].take() {
                    backprop_buffer[input] = Some(existing_gradient + epsilon);
                } else {
                    backprop_buffer[input] = Some(epsilon);
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        derivable::{activation::LeakyRelu, loss::Euclidean, regularize::NeuraL0},
        layer::dense::NeuraDenseLayer,
        utils::uniform_vector,
    };

    use super::*;

    #[test]
    fn test_graph_backprop() {
        let network =
            neura_sequential![neura_layer!("dense", 4, f64), neura_layer!("dense", 2, f64),]
                .construct(NeuraShape::Vector(10))
                .unwrap();

        let graph = NeuraGraph::from_sequential(network.clone(), NeuraShape::Vector(10));

        let trainer = NeuraGraphBackprop::new(Euclidean);

        let input = uniform_vector(10);
        let target = uniform_vector(2);

        let expected = NeuraBackprop::new(Euclidean).get_gradient(&network, &input, &target);
        let actual = trainer.get_gradient(&graph, &input, &target);

        type Gradient = <NeuraDenseLayer<f64, LeakyRelu<f64>, NeuraL0> as NeuraLayerBase>::Gradient;
        fn get_gradient(dynamic: &Box<dyn NeuraDynVectorSpace>) -> &Gradient {
            (**dynamic).into_any().downcast_ref::<Gradient>().unwrap()
        }

        assert_eq!(get_gradient(&actual[1]), &expected.0);
        assert_eq!(get_gradient(&actual[2]), &expected.1 .0);
    }
}
