use nalgebra::{DVector, Scalar};

use super::*;

/// **Class invariant:** start and end are
#[derive(Clone, Debug)]
pub struct NeuraIsolateLayer {
    start: NeuraShape,
    end: NeuraShape,
}

#[derive(Clone, Debug)]
pub enum NeuraIsolateLayerErr {
    Incompatible {
        start: NeuraShape,
        end: NeuraShape,
        input_shape: NeuraShape,
    },
    OutOfBound {
        start: NeuraShape,
        end: NeuraShape,
        input_shape: NeuraShape,
    },
    OutOfOrder {
        start: NeuraShape,
        end: NeuraShape,
    },
}

impl NeuraIsolateLayer {
    pub fn new<T: Into<NeuraShape>>(start: T, end: T) -> Option<Self> {
        let start = start.into();
        let end = end.into();

        if start.is_compatible(end) {
            Some(Self { start, end })
        } else {
            None
        }
    }
}

impl NeuraShapedLayer for NeuraIsolateLayer {
    fn output_shape(&self) -> NeuraShape {
        self.end.sub(self.start).unwrap_or_else(|| unreachable!())
    }
}

impl NeuraPartialLayer for NeuraIsolateLayer {
    type Constructed = NeuraIsolateLayer;
    type Err = NeuraIsolateLayerErr;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        use NeuraShape::*;
        let start = self.start;
        let end = self.end;

        match (input_shape, start, end) {
            (Vector(xi), Vector(xs), Vector(xe)) => {
                if xs >= xe {
                    return Err(NeuraIsolateLayerErr::OutOfOrder { start, end });
                }

                if xs >= xi || xe > xi {
                    return Err(NeuraIsolateLayerErr::OutOfBound {
                        start,
                        end,
                        input_shape,
                    });
                }

                Ok(self)
            }

            (Matrix(_xi, _yi), Matrix(_xs, _ys), Matrix(_xe, _ye)) => unimplemented!(),
            (Tensor(_xi, _yi, _zi), Tensor(_xs, _ys, _zs), Tensor(_xe, _ye, _ze)) => {
                unimplemented!()
            }

            _ => Err(NeuraIsolateLayerErr::Incompatible {
                start,
                end,
                input_shape,
            }),
        }
    }
}

impl<F: Clone + Scalar> NeuraLayer<DVector<F>> for NeuraIsolateLayer {
    type Output = DVector<F>;

    fn eval(&self, input: &DVector<F>) -> Self::Output {
        let (NeuraShape::Vector(start), NeuraShape::Vector(end)) = (self.start, self.end) else {
            panic!("NeuraIsolateLayer expected a value of dimension {}, got a vector", self.start.dims());
        };

        DVector::from_iterator(end - start, input.iter().cloned().skip(start).take(end))
    }
}

impl NeuraTrainableLayerBase for NeuraIsolateLayer {
    type Gradient = ();

    #[inline(always)]
    fn default_gradient(&self) -> Self::Gradient {
        ()
    }

    #[inline(always)]
    fn apply_gradient(&mut self, _gradient: &Self::Gradient) {
        // Noop
    }
}

impl<F: Clone + Scalar> NeuraTrainableLayerEval<DVector<F>> for NeuraIsolateLayer {
    type IntermediaryRepr = ();

    fn eval_training(&self, input: &DVector<F>) -> (Self::Output, Self::IntermediaryRepr) {
        (self.eval(input), ())
    }
}

impl<Input> NeuraTrainableLayerSelf<Input> for NeuraIsolateLayer
where
    Self: NeuraTrainableLayerEval<Input>,
{
    #[inline(always)]
    fn regularize_layer(&self) -> Self::Gradient {
        ()
    }

    #[inline(always)]
    fn get_gradient(
        &self,
        _input: &Input,
        _intermediary: &Self::IntermediaryRepr,
        _epsilon: &Self::Output,
    ) -> Self::Gradient {
        ()
    }
}

impl<F: Clone + Scalar + Default> NeuraTrainableLayerBackprop<DVector<F>> for NeuraIsolateLayer {
    fn backprop_layer(
        &self,
        input: &DVector<F>,
        _intermediary: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> DVector<F> {
        let mut result = DVector::from_element(input.len(), F::default());
        let NeuraShape::Vector(start) = self.start else {
            unreachable!();
        };

        for i in 0..epsilon.len() {
            result[start + i] = epsilon[i].clone();
        }

        result
    }
}
