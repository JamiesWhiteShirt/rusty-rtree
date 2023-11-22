use std::{fmt::Debug, marker::PhantomData, ops::Sub, ptr};

use crate::{
    bounds::{empty_bounds, min_bounds, min_bounds_all, Bounded, Bounds},
    fc_vec::{self, FCVec, FCVecContainer, FCVecOps},
    intersects::Intersects,
    select, split,
    util::{get_only, GetOnlyResult},
};

pub(crate) enum NodeEntry<'a, N, const D: usize, Key, Value> {
    Inner(NodeContainer<'a, N, D, Key, Value>),
    Leaf((Key, Value)),
}

impl<'a, N, const D: usize, Key, Value> Bounded<N, D> for NodeEntry<'a, N, D, Key, Value>
where
    N: Clone,
    Key: Bounded<N, D>,
{
    fn bounds(&self) -> Bounds<N, D> {
        match self {
            NodeEntry::Inner(node_container) => node_container.node.bounds(),
            NodeEntry::Leaf((key, _)) => key.bounds(),
        }
    }
}

pub(crate) struct Node<N, const D: usize, Key, Value> {
    pub(crate) bounds: Bounds<N, D>,
    children: FCVec,

    _phantom: PhantomData<(Key, Value)>,
}

impl<N, const D: usize, Key, Value> Node<N, D, Key, Value> {
    unsafe fn new(bounds: Bounds<N, D>, children: FCVec, _level: usize) -> Self {
        Node {
            bounds,
            children,

            _phantom: PhantomData,
        }
    }

    pub(crate) unsafe fn children(&self) -> &FCVec {
        return &self.children;
    }

    pub(crate) unsafe fn children_mut(&mut self) -> &mut FCVec {
        return &mut self.children;
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
    pub(crate) unsafe fn debug_assert_bvh(&self, level: usize) -> Bounds<N, D>
    where
        Key: Bounded<N, D>,
        N: Ord + num_traits::Bounded + Clone + Eq + Debug,
    {
        let bounds = if level > 0 {
            min_bounds_all(
                fc_vec::Iter::<Node<N, D, Key, Value>>::new(&self.children)
                    .map(|node| node.debug_assert_bvh(level - 1)),
            )
        } else {
            min_bounds_all(
                fc_vec::Iter::<(Key, Value)>::new(&self.children).map(|(key, _)| key.bounds()),
            )
        };
        assert_eq!(self.bounds, bounds);
        bounds
    }

    pub(crate) unsafe fn debug_assert_eq(a: &Self, b: &Self, level: usize)
    where
        N: Debug + Eq,
        Key: Debug + Eq,
        Value: Debug + Eq,
    {
        assert_eq!(a.bounds, b.bounds);
        if level > 0 {
            let ops = FCVecOps::<Node<N, D, Key, Value>>::new_ops(usize::MAX);
            let a_children = ops.as_slice(&b.children);
            let b_children = ops.as_slice(&b.children);
            assert_eq!(a_children.len(), b_children.len());
            for (a_child, b_child) in a_children.iter().zip(b_children.iter()) {
                Self::debug_assert_eq(a_child, b_child, level - 1);
            }
        } else {
            let ops = FCVecOps::<(Key, Value)>::new_ops(usize::MAX);
            let self_children = ops.as_slice(&a.children);
            let other_children = ops.as_slice(&b.children);
            assert_eq!(self_children, other_children);
        }
    }

    pub(crate) unsafe fn debug_assert_min_children(
        &self,
        level: usize,
        min_children: usize,
        is_root: bool,
    ) {
        if !is_root {
            assert!(self.children.len() >= min_children);
        }
        if level > 0 {
            let ops = FCVecOps::<Node<N, D, Key, Value>>::new_ops(usize::MAX);
            let children = ops.as_slice(&self.children);
            for child in children {
                child.debug_assert_min_children(level - 1, min_children, false);
            }
        }
    }
}

pub(crate) struct NodeOps<N, const D: usize, Key, Value> {
    leaf: FCVecOps<(Key, Value)>,
    inner: FCVecOps<Node<N, D, Key, Value>>,
}

impl<N, const D: usize, Key, Value> NodeOps<N, D, Key, Value> {
    pub(crate) fn new_ops(cap: usize) -> Self {
        NodeOps {
            leaf: FCVecOps::new_ops(cap),
            inner: FCVecOps::new_ops(cap),
        }
    }

    pub(crate) unsafe fn emtpy_leaf(&self) -> Node<N, D, Key, Value>
    where
        N: num_traits::Bounded,
    {
        Node::new(empty_bounds(), self.leaf.new(), 0)
    }

    pub(crate) unsafe fn empty_inner(&self, level: usize) -> Node<N, D, Key, Value>
    where
        N: num_traits::Bounded,
    {
        Node::new(empty_bounds(), self.inner.new(), level)
    }

    pub(crate) unsafe fn take_leaf_children(
        &self,
        node: Node<N, D, Key, Value>,
    ) -> FCVecContainer<(Key, Value)> {
        self.leaf.wrap(node.children)
    }

    pub(crate) unsafe fn take_inner_children(
        &self,
        node: Node<N, D, Key, Value>,
    ) -> FCVecContainer<Node<N, D, Key, Value>> {
        self.inner.wrap(node.children)
    }

    /*
    Branches the root node into two nodes in-place, such that the previous root
    node and the sibling node become children of the new root node.
     */
    pub(crate) unsafe fn branch(
        &self,
        root: &mut Node<N, D, Key, Value>,
        level: usize,
        sibling: Node<N, D, Key, Value>,
    ) where
        N: Ord + Clone + Sub<Output = N> + Into<f64>,
    {
        let bounds = min_bounds(&root.bounds, &sibling.bounds);
        let mut next_root_children = self.inner.new();
        self.inner.push(&mut next_root_children, ptr::read(root));
        self.inner.push(&mut next_root_children, sibling);
        ptr::write(root, Node::new(bounds, next_root_children, level));
    }

    pub(crate) unsafe fn take_single_inner_child(
        &self,
        node: &mut Node<N, D, Key, Value>,
    ) -> Option<Node<N, D, Key, Value>>
    where
        N: num_traits::Bounded,
    {
        if node.children.len() == 1 {
            node.bounds = empty_bounds();
            Some(self.inner.remove(&mut node.children, 0))
        } else {
            None
        }
    }

    unsafe fn self_insert<'a>(
        &self,
        node: &'a mut Node<N, D, Key, Value>,
        level: usize,
        min_children: usize,
        entry: NodeEntry<'a, N, D, Key, Value>,
    ) -> Option<Node<N, D, Key, Value>>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D>,
    {
        let entry_bounds = entry.bounds();
        match entry {
            NodeEntry::Inner(node_container) => {
                if node.children.len() < self.inner.cap() {
                    self.inner.push(&mut node.children, node_container.unwrap());
                    node.bounds = min_bounds(&node.bounds, &entry_bounds);
                    None
                } else {
                    let (new_bounds, sibling_bounds, sibling_children) = split::quadratic_n(
                        min_children,
                        &self.inner,
                        node_container.unwrap(),
                        &mut node.children,
                    );
                    node.bounds = new_bounds;
                    Some(Node::new(sibling_bounds, sibling_children, level))
                }
            }
            NodeEntry::Leaf(entry) => {
                if node.children.len() < self.leaf.cap() {
                    self.leaf.push(&mut node.children, entry);
                    node.bounds = min_bounds(&node.bounds, &entry_bounds);
                    None
                } else {
                    let (new_bounds, sibling_bounds, sibling_children) =
                        split::quadratic_n(min_children, &self.leaf, entry, &mut node.children);
                    node.bounds = new_bounds;
                    Some(Node::new(sibling_bounds, sibling_children, level))
                }
            }
        }
    }

    pub(crate) unsafe fn insert<'a>(
        &self,
        node: &'a mut Node<N, D, Key, Value>,
        level: usize,
        min_children: usize,
        depth: usize,
        entry: NodeEntry<'a, N, D, Key, Value>,
    ) -> Option<Node<N, D, Key, Value>>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Bounded<N, D>,
    {
        let entry_bounds = entry.bounds();
        if depth > 0 {
            let mut insert_child = NodeRefMut::new(
                self,
                depth - 1,
                select::minimal_volume_increase(
                    self.inner.as_slice_mut(&mut node.children),
                    &entry.bounds(),
                )
                .unwrap(),
            );
            if let Some(new_child) = insert_child.insert(min_children, entry) {
                // The child node split, so the entries in new_child are no longer part of self
                // Recompute the bounds of self before trying to insert new_child into self
                node.bounds = min_bounds_all(
                    self.inner
                        .as_slice(&node.children)
                        .iter()
                        .map(|child| child.bounds()),
                );
                self.self_insert(node, level, min_children, NodeEntry::Inner(new_child))
            } else {
                node.bounds = min_bounds(&node.bounds, &entry_bounds);
                None
            }
        } else {
            self.self_insert(node, level, min_children, entry)
        }
    }

    pub(crate) unsafe fn remove(
        &self,
        node: &mut Node<N, D, Key, Value>,
        min_children: usize,
        level: usize,
        key: &Key,
        underfull_nodes: &mut [Option<Node<N, D, Key, Value>>],
    ) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq,
        Value: Eq,
    {
        if level > 0 {
            let mut i = node.children.len();
            while i > 0 {
                i -= 1;
                if self
                    .inner
                    .at(&node.children, i)
                    .bounds
                    .intersects(&key.bounds())
                {
                    if let Some(value) = self.remove(
                        self.inner.at_mut(&mut node.children, i),
                        min_children,
                        level - 1,
                        key,
                        underfull_nodes,
                    ) {
                        if self.inner.at(&node.children, i).children.len() < min_children {
                            let removed_child = self.inner.swap_remove(&mut node.children, i);
                            underfull_nodes[level - 1] = Some(removed_child);
                        }

                        node.bounds = min_bounds_all(
                            self.inner
                                .as_slice(&node.children)
                                .iter()
                                .map(|child| child.bounds()),
                        );

                        return Some(value);
                    }
                }
            }
            return None;
        } else {
            let index = self
                .leaf
                .as_slice(&node.children)
                .iter()
                .position(|(k, v)| k == key);
            if let Some(i) = index {
                let value = self.leaf.swap_remove(&mut node.children, i).1;
                node.bounds = min_bounds_all(
                    self.leaf
                        .as_slice(&node.children)
                        .iter()
                        .map(|(key, _)| key.bounds()),
                );

                return Some(value);
            }
            return None;
        }
    }

    pub(crate) unsafe fn drop(&self, node: &mut Node<N, D, Key, Value>, level: usize) {
        if level > 0 {
            for child in self.inner.as_slice_mut(&mut node.children) {
                self.drop(child, level - 1);
            }
            self.inner.drop(&mut node.children);
        } else {
            self.leaf.drop(&mut node.children);
        }
    }

    pub(crate) unsafe fn clone(
        &self,
        node: &Node<N, D, Key, Value>,
        level: usize,
    ) -> Node<N, D, Key, Value>
    where
        N: Clone,
        Key: Clone,
        Value: Clone,
    {
        if level > 0 {
            let mut clone_children = self.inner.new();
            for child in self.inner.as_slice(&node.children) {
                self.inner
                    .push(&mut clone_children, self.clone(child, level - 1));
            }
            Node::new(node.bounds.clone(), clone_children, level)
        } else {
            Node::new(node.bounds.clone(), self.leaf.clone(&node.children), level)
        }
    }

    pub(crate) unsafe fn len(&self, node: &Node<N, D, Key, Value>, level: usize) -> usize {
        if level > 0 {
            let mut size = 0;
            for child in self.inner.as_slice(&node.children) {
                size += self.len(child, level - 1);
            }
            size
        } else {
            node.children.len()
        }
    }

    pub(crate) unsafe fn get<'a>(
        &self,
        node: &'a Node<N, D, Key, Value>,
        level: usize,
        key: &Key,
    ) -> Option<&'a Value>
    where
        N: Ord,
        Key: Eq + Bounded<N, D>,
    {
        if level > 0 {
            for child in self.inner.as_slice(&node.children) {
                if child.bounds.contains(&key.bounds()) {
                    if let Some(value) = self.get(child, level - 1, key) {
                        return Some(value);
                    }
                }
            }
            None
        } else {
            self.leaf
                .as_slice(&node.children)
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
        }
    }

    pub(crate) unsafe fn get_mut<'a>(
        &self,
        node: &'a mut Node<N, D, Key, Value>,
        level: usize,
        key: &Key,
    ) -> Option<&'a mut Value>
    where
        N: Ord,
        Key: Eq + Bounded<N, D>,
    {
        if level > 0 {
            for child in self.inner.as_slice_mut(&mut node.children) {
                if child.bounds.contains(&key.bounds()) {
                    if let Some(value) = self.get_mut(child, level - 1, key) {
                        return Some(value);
                    }
                }
            }
            None
        } else {
            self.leaf
                .as_slice_mut(&mut node.children)
                .iter_mut()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
        }
    }

    pub(crate) unsafe fn insert_unique<'a>(
        &self,
        node: &'a mut Node<N, D, Key, Value>,
        level: usize,
        min_children: usize,
        key: Key,
        value: Value,
    ) -> (Option<Value>, Option<Node<N, D, Key, Value>>)
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Eq + Bounded<N, D>,
    {
        let entry_bounds = key.bounds();
        if level > 0 {
            // Check which children contain the key bounds
            let children_containing_key = self
                .inner
                .as_slice_mut(&mut node.children)
                .iter_mut()
                .filter(|child| child.bounds.contains(&entry_bounds));
            match get_only(children_containing_key) {
                GetOnlyResult::None => {
                    // If there are no children containing the key bounds, then
                    // the key cannot exist in the node.
                    (
                        None,
                        self.insert(
                            node,
                            level,
                            min_children,
                            level,
                            NodeEntry::Leaf((key, value)),
                        ),
                    )
                }
                GetOnlyResult::Only(child) => {
                    let mut child = NodeRefMut::new(self, level - 1, child);
                    // If there is only one children containing the key bounds,
                    // then we can maintain the uniqueness invariant by
                    // performing a unique insert into that child.
                    let (old_value, new_child) = child.insert_unique(min_children, key, value);

                    (
                        old_value,
                        if let Some(new_child) = new_child {
                            // The child node split, so the entries in new_child are no longer part of self
                            // Recompute the bounds of self before trying to insert new_child into self
                            node.bounds = min_bounds_all(
                                self.inner
                                    .as_slice(&node.children)
                                    .iter()
                                    .map(|child| child.bounds()),
                            );
                            self.self_insert(node, level, min_children, NodeEntry::Inner(new_child))
                        } else {
                            node.bounds = min_bounds(&node.bounds, &entry_bounds);
                            None
                        },
                    )
                }
                GetOnlyResult::Multiple => {
                    // Try to find the key among the children and overwrite it if it
                    // exists. If it doesn't exist, perform a regular insert.

                    if let Some(value_ref) = self.get_mut(node, level, &key) {
                        (Some(std::mem::replace(value_ref, value)), None)
                    } else {
                        (
                            None,
                            self.insert(
                                node,
                                level,
                                min_children,
                                level,
                                NodeEntry::Leaf((key, value)),
                            ),
                        )
                    }
                }
            }
        } else {
            // Try to find the key among the children and overwrite it if it
            // exists. If it doesn't exist, perform a regular insert.
            if let Some(entry) = self
                .leaf
                .as_slice_mut(&mut node.children)
                .iter_mut()
                .find(|(k, _)| k == &key)
            {
                (Some(std::mem::replace(&mut entry.1, value)), None)
            } else {
                (
                    None,
                    self.insert(
                        node,
                        level,
                        min_children,
                        level,
                        NodeEntry::Leaf((key, value)),
                    ),
                )
            }
        }
    }
}

pub(crate) struct NodeRefMut<'a, 'b, N, const D: usize, Key, Value> {
    ops: &'a NodeOps<N, D, Key, Value>,
    level: usize,
    node: &'b mut Node<N, D, Key, Value>,
}

impl<'a, 'b, N, const D: usize, Key, Value> NodeRefMut<'a, 'b, N, D, Key, Value> {
    pub(crate) unsafe fn new(
        ops: &'a NodeOps<N, D, Key, Value>,
        level: usize,
        node: &'b mut Node<N, D, Key, Value>,
    ) -> Self {
        NodeRefMut {
            ops,
            level,
            node: node,
        }
    }

    unsafe fn take_single_inner_child(&mut self) -> Option<NodeContainer<N, D, Key, Value>>
    where
        N: num_traits::Bounded,
    {
        unsafe {
            self.ops
                .take_single_inner_child(&mut self.node)
                .map(|child| NodeContainer::new(self.ops, self.level - 1, child))
        }
    }

    unsafe fn insert(
        &mut self,
        min_children: usize,
        entry: NodeEntry<'a, N, D, Key, Value>,
    ) -> Option<NodeContainer<'a, N, D, Key, Value>>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Bounded<N, D>,
    {
        unsafe {
            self.ops
                .insert(&mut self.node, self.level, min_children, self.level, entry)
                .map(|node| NodeContainer::new(self.ops, self.level, node))
        }
    }

    fn remove(
        &mut self,
        min_children: usize,
        key: &Key,
        underfull_nodes: &mut [Option<Node<N, D, Key, Value>>],
    ) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq,
        Value: Eq,
    {
        unsafe {
            self.ops.remove(
                &mut self.node,
                min_children,
                self.level,
                key,
                underfull_nodes,
            )
        }
    }

    pub(crate) fn get_mut(self, key: &Key) -> Option<&'b mut Value>
    where
        N: Ord,
        Key: Eq + Bounded<N, D>,
    {
        unsafe { self.ops.get_mut(self.node, self.level, key) }
    }

    pub(crate) fn insert_unique(
        &mut self,
        min_children: usize,
        key: Key,
        value: Value,
    ) -> (Option<Value>, Option<NodeContainer<'a, N, D, Key, Value>>)
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Eq + Bounded<N, D>,
    {
        unsafe {
            let (prev_value, new_sibling) =
                self.ops
                    .insert_unique(&mut self.node, self.level, min_children, key, value);
            (
                prev_value,
                new_sibling.map(|node| NodeContainer::new(self.ops, self.level, node)),
            )
        }
    }
}

pub(crate) struct NodeContainer<'a, N, const D: usize, Key, Value> {
    ops: &'a NodeOps<N, D, Key, Value>,
    level: usize,
    node: Node<N, D, Key, Value>,
}

impl<'a, N, const D: usize, Key, Value> NodeContainer<'a, N, D, Key, Value> {
    pub(crate) unsafe fn new(
        ops: &'a NodeOps<N, D, Key, Value>,
        level: usize,
        node: Node<N, D, Key, Value>,
    ) -> Self {
        NodeContainer { ops, level, node }
    }

    unsafe fn take_single_inner_child(&mut self) -> Option<NodeContainer<N, D, Key, Value>>
    where
        N: num_traits::Bounded,
    {
        unsafe {
            self.ops
                .take_single_inner_child(&mut self.node)
                .map(|child| NodeContainer::new(self.ops, self.level - 1, child))
        }
    }

    fn insert(
        &mut self,
        min_children: usize,
        entry: NodeEntry<N, D, Key, Value>,
    ) -> Option<NodeContainer<'a, N, D, Key, Value>>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Bounded<N, D>,
    {
        unsafe {
            self.ops
                .insert(&mut self.node, self.level, min_children, self.level, entry)
                .map(|node| NodeContainer::new(self.ops, self.level, node))
        }
    }

    fn remove(
        &mut self,
        min_children: usize,
        key: &Key,
        underfully_nodes: &mut [Option<Node<N, D, Key, Value>>],
    ) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq,
        Value: Eq,
    {
        unsafe {
            self.ops.remove(
                &mut self.node,
                min_children,
                self.level,
                key,
                underfully_nodes,
            )
        }
    }

    pub(crate) fn get_ref(&self) -> NodeRef<N, D, Key, Value> {
        unsafe { NodeRef::new(self.ops, self.level, &self.node) }
    }

    pub(crate) fn get_ref_mut(&mut self) -> NodeRefMut<N, D, Key, Value> {
        unsafe { NodeRefMut::new(self.ops, self.level, &mut self.node) }
    }

    /**
    Unwraps the node from the container without dropping it.
    */
    pub(crate) unsafe fn unwrap(self) -> Node<N, D, Key, Value> {
        let s = std::mem::ManuallyDrop::new(self);
        std::ptr::read(&s.node)
    }
}

impl<'a, N, const D: usize, Key, Value> Drop for NodeContainer<'a, N, D, Key, Value> {
    fn drop(&mut self) {
        unsafe { self.ops.drop(&mut self.node, self.level) }
    }
}

impl<'a, N, const D: usize, Key, Value> Clone for NodeContainer<'a, N, D, Key, Value>
where
    N: Clone,
    Key: Clone,
    Value: Clone,
{
    fn clone(&self) -> Self {
        Self {
            ops: self.ops,
            level: self.level,
            node: unsafe { self.ops.clone(&self.node, self.level) },
        }
    }
}

impl<'a, N, const D: usize, Key, Value> Debug for NodeContainer<'a, N, D, Key, Value>
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
                    ops: self.ops,
                    level: self.level,
                    children: &self.node.children,

                    _phantom: PhantomData,
                },
            )
            .finish()
    }
}

pub(crate) struct NodeRef<'a, 'b, N, const D: usize, Key, Value> {
    ops: &'a NodeOps<N, D, Key, Value>,
    level: usize,
    node: &'b Node<N, D, Key, Value>,
}

impl<'a, 'b, N, const D: usize, Key, Value> NodeRef<'a, 'b, N, D, Key, Value> {
    pub(crate) unsafe fn new(
        ops: &'a NodeOps<N, D, Key, Value>,
        level: usize,
        node: &'b Node<N, D, Key, Value>,
    ) -> Self {
        NodeRef { ops, level, node }
    }

    pub(crate) fn len(&self) -> usize {
        unsafe { self.ops.len(self.node, self.level) }
    }

    pub(crate) fn get(&self, key: &Key) -> Option<&'b Value>
    where
        N: Ord,
        Key: Eq + Bounded<N, D>,
    {
        unsafe { self.ops.get(self.node, self.level, key) }
    }
}

impl<'a, 'b, N, const D: usize, Key, Value> Debug for NodeRef<'a, 'b, N, D, Key, Value>
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
                    ops: self.ops,
                    level: self.level,
                    children: &self.node.children,

                    _phantom: PhantomData,
                },
            )
            .finish()
    }
}

pub(crate) struct NodeChildrenRef<'a, 'b, N, const D: usize, Key, Value> {
    ops: &'a NodeOps<N, D, Key, Value>,
    level: usize,
    children: &'b FCVec,

    _phantom: PhantomData<&'b (N, Key, Value)>,
}

impl<'a, 'b, N, const D: usize, Key, Value> Debug for NodeChildrenRef<'a, 'b, N, D, Key, Value>
where
    N: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.level > 0 {
            f.debug_list()
                .entries(unsafe {
                    fc_vec::Iter::<Node<N, D, Key, Value>>::new(self.children).map(|node| NodeRef {
                        ops: self.ops,
                        level: self.level - 1,
                        node,
                    })
                })
                .finish()
        } else {
            f.debug_list()
                .entries(unsafe {
                    fc_vec::Iter::<(Key, Value)>::new(self.children)
                        .map(|(key, value)| (key, value))
                })
                .finish()
        }
    }
}
