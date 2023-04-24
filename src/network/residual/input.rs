use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct NeuraResidualInput<T> {
    // TODO: try to remove this Rc
    slots: Vec<Vec<Rc<T>>>,
}

impl<T> NeuraResidualInput<T> {
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    pub fn push(&mut self, offset: usize, data: Rc<T>) {
        while self.slots.len() <= offset {
            self.slots.push(Vec::new());
        }
        self.slots[offset].push(data);
    }

    pub fn shift(&self) -> (Vec<Rc<T>>, NeuraResidualInput<T>) {
        let res = self.slots.get(0).cloned().unwrap_or(vec![]);
        let new_input = Self {
            slots: self.slots.iter().skip(1).cloned().collect(),
        };

        (res, new_input)
    }

    /// Returns the first input item of the first slot
    pub fn get_first(self) -> Option<Rc<T>> {
        // TODO: return None if the first slot is bigger than 1 or if there are multiple non-empty slots
        self.slots
            .into_iter()
            .next()
            .and_then(|first_slot| first_slot.into_iter().next())
    }
}
