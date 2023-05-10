use super::*;

/// Last element of a NeuraSequential network
#[derive(Clone, Debug, PartialEq, Copy)]
#[derive(Default)]
pub struct NeuraSequentialLast {
    shape: Option<NeuraShape>,
}

impl NeuraPartialLayer for NeuraSequentialLast {
    type Constructed = NeuraSequentialLast;

    type Err = ();

    fn construct(mut self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        self.shape = Some(input_shape);
        Ok(self)
    }
}

impl NeuraLayerBase for NeuraSequentialLast {
    type Gradient = ();

    #[inline(always)]
    fn output_shape(&self) -> NeuraShape {
        self.shape
            .expect("Called NeuraSequentialLast::output_shape() without building it")
    }

    #[inline(always)]
    fn default_gradient(&self) -> Self::Gradient {
        
    }
}

impl<Input: Clone> NeuraLayer<Input> for NeuraSequentialLast {
    type Output = Input;
    type IntermediaryRepr = ();

    #[inline(always)]
    fn eval_training(&self, input: &Input) -> (Self::Output, Self::IntermediaryRepr) {
        (input.clone(), ())
    }

    #[inline(always)]
    fn backprop_layer(
        &self,
        _input: &Input,
        _intermediary: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> Input {
        epsilon.clone()
    }
}

impl NeuraNetworkBase for NeuraSequentialLast {
    type Layer = ();

    #[inline(always)]
    fn get_layer(&self) -> &Self::Layer {
        &()
    }
}

impl NeuraNetworkRec for NeuraSequentialLast {
    type NextNode = ();

    #[inline(always)]
    fn get_next(&self) -> &Self::NextNode {
        &()
    }

    #[inline(always)]
    fn merge_gradient(
        &self,
        rec_gradient: <Self::NextNode as NeuraLayerBase>::Gradient,
        _layer_gradient: <Self::Layer as NeuraLayerBase>::Gradient,
    ) -> Self::Gradient
    where
        Self::Layer: NeuraLayerBase,
    {
        rec_gradient
    }
}

impl<Input: Clone> NeuraNetwork<Input> for NeuraSequentialLast {
    type LayerInput = Input;
    type NodeOutput = Input;

    fn map_input<'a>(&'_ self, input: &'a Input) -> Cow<'a, Self::LayerInput> {
        Cow::Borrowed(input)
    }

    fn map_output<'a>(
        &'_ self,
        _input: &'_ Input,
        layer_output: &'a Input,
    ) -> Cow<'a, Self::NodeOutput> {
        Cow::Borrowed(layer_output)
    }

    fn map_gradient_in<'a>(
        &'_ self,
        _input: &'_ Input,
        gradient_in: &'a Self::NodeOutput,
    ) -> Cow<'a, Input> {
        Cow::Borrowed(gradient_in)
    }

    fn map_gradient_out<'a>(
        &'_ self,
        _input: &'_ Input,
        _gradient_in: &'_ Self::NodeOutput,
        gradient_out: &'a Self::LayerInput,
    ) -> Cow<'a, Input> {
        Cow::Borrowed(gradient_out)
    }
}



/// Operations on the tail end of a sequential network
pub trait NeuraSequentialTail {
    type TailTrimmed;
    type TailPushed<T>;

    fn trim_tail(self) -> Self::TailTrimmed;
    fn push_tail<T>(self, layer: T) -> Self::TailPushed<T>;
}

// Trimming the last layer returns an empty network
impl<Layer> NeuraSequentialTail for NeuraSequential<Layer, NeuraSequentialLast> {
    type TailTrimmed = NeuraSequentialLast;
    // GAT :3
    type TailPushed<T> = NeuraSequential<Layer, NeuraSequential<T, NeuraSequentialLast>>;

    fn trim_tail(self) -> Self::TailTrimmed {
        NeuraSequentialLast::default()
    }

    fn push_tail<T>(self, layer: T) -> Self::TailPushed<T> {
        NeuraSequential {
            layer: self.layer,
            child_network: Box::new(NeuraSequential {
                layer,
                child_network: Box::<NeuraSequentialLast>::default(),
            }),
        }
    }
}

// Trimming another layer returns a network which calls trim recursively
impl<Layer, ChildNetwork: NeuraSequentialTail> NeuraSequentialTail
    for NeuraSequential<Layer, ChildNetwork>
{
    type TailTrimmed = NeuraSequential<Layer, <ChildNetwork as NeuraSequentialTail>::TailTrimmed>;
    type TailPushed<T> =
        NeuraSequential<Layer, <ChildNetwork as NeuraSequentialTail>::TailPushed<T>>;

    fn trim_tail(self) -> Self::TailTrimmed {
        NeuraSequential {
            layer: self.layer,
            child_network: Box::new(self.child_network.trim_tail()),
        }
    }

    fn push_tail<T>(self, layer: T) -> Self::TailPushed<T> {
        NeuraSequential {
            layer: self.layer,
            child_network: Box::new(self.child_network.push_tail(layer)),
        }
    }
}
