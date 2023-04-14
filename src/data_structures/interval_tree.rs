use std::cmp::max;
use std::ops::{Deref, Range, RangeInclusive};

/// Interval Trees are used to store intervals and answer range queries like "how do intervals overlap with each other"
/// For instance say we have the following set of intervals `[5..=20, 10..=30, 12..=15, 15..=20, 17..=19, 30..=40]` and we want to answer a query like "which intervals does [6..=8] overlaps with?"
/// The idea behind Interval Trees is to store, in addition

#[derive(Debug, Clone)]
struct IntervalNode<T: Ord + PartialEq + Eq + Clone + Copy> {
    interval: RangeInclusive<T>,
    max_left: T,
    left: Option<Box<IntervalNode<T>>>,
    right: Option<Box<IntervalNode<T>>>,
}

impl<T: Ord + PartialEq + Eq + Clone + Copy> IntervalNode<T> {
    fn new(interval: RangeInclusive<T>) -> Self {
        let max = *interval.end();
        Self {
            interval,
            max_left: max,
            left: None,
            right: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct IntervalTree<T: Ord + PartialEq + Eq + Clone + Copy> {
    root: Option<Box<IntervalNode<T>>>,
}

impl<T: Ord + PartialEq + Eq + Clone + Copy> IntervalTree<T> {
    pub fn insert(&mut self, range: RangeInclusive<T>) {
        self.root = IntervalTree::insert_under_rec(&mut self.root, range);
    }

    pub fn traverse_inorder(&self) -> Vec<RangeInclusive<T>> {
        match &self.root {
            None => Vec::default(),
            Some(root) => {
                let mut collector = Vec::default();
                IntervalTree::traverse_inorder_rec(root, &mut collector);
                collector
            }
        }
    }

    fn traverse_inorder_rec(node: &Box<IntervalNode<T>>, collector: &mut Vec<RangeInclusive<T>>) {
        if let Some(left) = &node.left {
            IntervalTree::traverse_inorder_rec(left, collector);
        }
        collector.push(node.interval.clone());
        if let Some(right) = &node.right {
            IntervalTree::traverse_inorder_rec(right, collector);
        }
    }

    fn insert_under_rec(
        under: &mut Option<Box<IntervalNode<T>>>,
        new: RangeInclusive<T>,
    ) -> Option<Box<IntervalNode<T>>> {
        match under {
            None => Some(Box::new(IntervalNode::new(new))),
            Some(value) => {
                let old_max = value.max_left;
                let new_node = if new.start() < value.interval.start() {
                    // insert left
                    value.left = IntervalTree::insert_under_rec(&mut value.left, new);
                    &value.left
                } else {
                    value.right = IntervalTree::insert_under_rec(&mut value.right, new);
                    &value.right
                };
                let new_max = new_node.as_ref().unwrap().max_left;
                value.max_left = max(new_max, old_max);
                Some(value.clone())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::data_structures::interval_tree::IntervalTree;

    #[test]
    fn test_sample_inorder_traversal() {
        let mut tree: IntervalTree<i32> = IntervalTree::default();
        tree.insert(15..=20);
        tree.insert(10..=30);
        tree.insert(17..=19);
        tree.insert(30..=40);
        tree.insert(5..=20);
        tree.insert(12..=15);
        let sorted_intervals = tree.traverse_inorder();
        assert_eq!(
            sorted_intervals,
            vec![5..=20, 10..=30, 12..=15, 15..=20, 17..=19, 30..=40]
        );
    }
}
