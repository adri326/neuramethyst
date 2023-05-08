use nalgebra::{DVector, Scalar};

use crate::err::NeuraIsolateLayerErr;

use super::*;

/// **Class invariant:** start and end are
#[derive(Clone, Debug)]
pub struct NeuraIsolateLayer {
    start: NeuraShape,
    end: NeuraShape,
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

impl NeuraLayerBase for NeuraIsolateLayer {
    type Gradient = ();

    #[inline(always)]
    fn default_gradient(&self) -> Self::Gradient {
        ()
    }

    fn output_shape(&self) -> NeuraShape {
        self.end.sub(self.start).unwrap_or_else(|| unreachable!())
    }
}

impl<F: Clone + Scalar + Default> NeuraLayer<DVector<F>> for NeuraIsolateLayer {
    type Output = DVector<F>;

    type IntermediaryRepr = ();

    fn eval_training(&self, input: &DVector<F>) -> (Self::Output, Self::IntermediaryRepr) {
        let (NeuraShape::Vector(start), NeuraShape::Vector(end)) = (self.start, self.end) else {
            panic!("NeuraIsolateLayer expected a value of dimension {}, got a vector", self.start.dims());
        };

        let res = DVector::from_iterator(end - start, input.iter().cloned().skip(start).take(end));

        (res, ())
    }

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
