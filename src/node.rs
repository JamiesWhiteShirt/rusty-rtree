use std::{fmt::Debug, marker::PhantomData, ops::Sub};

use crate::{
    bounds::{empty_bounds, min_bounds, min_bounds_all, Bounded, Bounds},
    fs_vec::{self, FSVecData, FSVecOps},
    intersects::Intersects,
    select, split,
};

pub(crate) enum NodeEntry<N, const D: usize, Key, Value> {
    Inner(Node<N, D, Key, Value>),
    Leaf((Key, Value)),
}

impl<N, const D: usize, Key, Value> Bounded<N, D> for NodeEntry<N, D, Key, Value>
where
    N: Clone,
    Key: Bounded<N, D>,
{
    fn bounds(&self) -> Bounds<N, D> {
        match self {
            NodeEntry::Inner(node_ref) => node_ref.bounds(),
            NodeEntry::Leaf((key, _)) => key.bounds(),
        }
    }
}

pub(crate) struct Node<N, const D: usize, Key, Value> {
    pub(crate) bounds: Bounds<N, D>,
    pub(crate) children: FSVecData,

    _phantom: PhantomData<(Key, Value)>,
}

impl<N, const D: usize, Key, Value> Node<N, D, Key, Value> {
    pub(crate) fn new(bounds: Bounds<N, D>, children: FSVecData) -> Self {
        Node {
            bounds,
            children,

            _phantom: PhantomData,
        }
    }
}

impl<N, const D: usize, Key, Value> Bounded<N, D> for (Key, Value)
where
    N: Clone,
    Key: Bounded<N, D>,
{
    fn bounds(&self) -> Bounds<N, D> {
        self.0.bounds()
    }
}

impl<N, const D: usize, Key, Value> Bounded<N, D> for Node<N, D, Key, Value>
where
    N: Clone,
{
    fn bounds(&self) -> Bounds<N, D> {
        self.bounds.clone()
    }
}

impl<N, const D: usize, Key, Value> Node<N, D, Key, Value> {
    pub(crate) unsafe fn drop(&mut self, level: usize, max_children: usize) {
        if level > 0 {
            let ops = FSVecOps::<Node<N, D, Key, Value>>::new_ops(max_children);
            for child in ops.as_slice_mut(&mut self.children) {
                child.drop(level - 1, max_children);
            }
            ops.drop(&mut self.children);
        } else {
            let ops = FSVecOps::<(Key, Value)>::new_ops(max_children);
            ops.drop(&mut self.children);
        }
    }

    pub(crate) unsafe fn take_single_inner_child(
        &mut self,
        max_children: usize,
    ) -> Option<Node<N, D, Key, Value>>
    where
        N: num_traits::Bounded,
    {
        if self.children.len() == 1 {
            let ops = FSVecOps::<Node<N, D, Key, Value>>::new_ops(max_children);
            self.bounds = empty_bounds();
            Some(ops.remove(&mut self.children, 0))
        } else {
            None
        }
    }

    unsafe fn insert_self_entry<T>(
        &mut self,
        min_children: usize,
        ops: FSVecOps<T>,
        entry: T,
    ) -> Option<Self>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64>,
        T: Bounded<N, D>,
    {
        let entry_bounds = entry.bounds();
        if self.children.len() < ops.cap() {
            ops.push(&mut self.children, entry);
            self.bounds = min_bounds(&self.bounds, &entry_bounds);
            None
        } else {
            let (self_bounds, new_bounds, new_children) =
                split::quadratic_n(min_children, ops, entry, &mut self.children);
            self.bounds = self_bounds;
            Some(Node {
                bounds: new_bounds,
                children: new_children,

                _phantom: PhantomData,
            })
        }
    }

    pub(crate) unsafe fn insert_entry(
        &mut self,
        max_children: usize,
        min_children: usize,
        depth: usize,
        entry: NodeEntry<N, D, Key, Value>,
    ) -> Option<Node<N, D, Key, Value>>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D>,
    {
        if depth > 0 {
            let ops = FSVecOps::<Node<N, D, Key, Value>>::new_ops(max_children);

            let insert_child = select::minimal_volume_increase(
                ops.as_slice_mut(&mut self.children),
                &entry.bounds(),
            )
            .unwrap();
            if let Some(new_node_ref) =
                insert_child.insert_entry(max_children, min_children, depth - 1, entry)
            {
                self.insert_self_entry(min_children, ops, new_node_ref)
            } else {
                self.bounds = min_bounds(&self.bounds, &insert_child.bounds);
                None
            }
        } else {
            match entry {
                NodeEntry::Inner(entry) => {
                    self.insert_self_entry(min_children, FSVecOps::new_ops(max_children), entry)
                }
                NodeEntry::Leaf(entry) => {
                    self.insert_self_entry(min_children, FSVecOps::new_ops(max_children), entry)
                }
            }
        }
    }

    pub(crate) unsafe fn remove(
        &mut self,
        max_children: usize,
        min_children: usize,
        level: usize,
        key: &Key,
        value: &Value,
        reinsert_nodes: &mut [Option<FSVecData>],
    ) -> bool
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq,
        Value: Eq,
    {
        if level > 0 {
            let ops = FSVecOps::<Node<N, D, Key, Value>>::new_ops(max_children);
            let mut i = self.children.len();
            while i > 0 {
                i -= 1;
                if ops.at(&self.children, i).bounds.intersects(&key.bounds())
                    && ops.at_mut(&mut self.children, i).remove(
                        max_children,
                        min_children,
                        level - 1,
                        key,
                        value,
                        reinsert_nodes,
                    )
                {
                    if ops.at(&self.children, i).children.len() < min_children {
                        let removed_child = ops.swap_remove(&mut self.children, i);
                        reinsert_nodes[level - 1] = Some(removed_child.children);
                    }

                    self.bounds = min_bounds_all(
                        ops.as_slice(&self.children)
                            .iter()
                            .map(|child| child.bounds()),
                    );

                    return true;
                }
            }
            return false;
        } else {
            let ops = FSVecOps::<(Key, Value)>::new_ops(max_children);
            let index = ops
                .as_slice(&self.children)
                .iter()
                .position(|(k, v)| k == key && v == value);
            if let Some(i) = index {
                ops.swap_remove(&mut self.children, i);
                self.bounds = min_bounds_all(
                    ops.as_slice(&self.children)
                        .iter()
                        .map(|(key, _)| key.bounds()),
                );

                return true;
            }
            return false;
        }
    }

    pub(crate) unsafe fn debug_assert_bvh(&self, level: usize) -> Bounds<N, D>
    where
        Key: Bounded<N, D>,
        N: Ord + num_traits::Bounded + Clone + Eq + Debug,
    {
        let bounds = if level > 0 {
            min_bounds_all(
                fs_vec::Iter::<Node<N, D, Key, Value>>::new(&self.children)
                    .map(|node| node.debug_assert_bvh(level - 1)),
            )
        } else {
            min_bounds_all(
                fs_vec::Iter::<(Key, Value)>::new(&self.children).map(|(key, _)| key.bounds()),
            )
        };
        assert_eq!(bounds, self.bounds);
        bounds
    }
}

pub(crate) struct NodeRef<'a, N, const D: usize, Key, Value> {
    level: usize,
    node: &'a Node<N, D, Key, Value>,
}

impl<'a, N, const D: usize, Key, Value> NodeRef<'a, N, D, Key, Value> {
    pub(crate) fn new(level: usize, node: &'a Node<N, D, Key, Value>) -> Self {
        NodeRef { level, node }
    }
}

impl<'a, N, const D: usize, Key, Value> Debug for NodeRef<'a, N, D, Key, Value>
where
    N: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("bounds", &self.node.bounds)
            .field(
                "node",
                &NodeChildrenRef::<N, D, Key, Value> {
                    level: self.level,
                    children: &self.node.children,

                    _phantom: PhantomData,
                },
            )
            .finish()
    }
}

pub(crate) struct NodeChildrenRef<'a, N, const D: usize, Key, Value> {
    level: usize,
    children: &'a FSVecData,

    _phantom: PhantomData<&'a (N, Key, Value)>,
}

impl<'a, N, const D: usize, Key, Value> Debug for NodeChildrenRef<'a, N, D, Key, Value>
where
    N: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.level > 0 {
            f.debug_list()
                .entries(unsafe {
                    fs_vec::Iter::<Node<N, D, Key, Value>>::new(self.children).map(|node| NodeRef {
                        level: self.level - 1,
                        node,
                    })
                })
                .finish()
        } else {
            f.debug_list()
                .entries(unsafe {
                    fs_vec::Iter::<(Key, Value)>::new(self.children)
                        .map(|(key, value)| (key, value))
                })
                .finish()
        }
    }
}
