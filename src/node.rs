use std::{
    borrow::Borrow,
    fmt::Debug,
    mem::{self, ManuallyDrop},
    num::NonZeroUsize,
    ops::Sub,
    ptr, slice,
};

use crate::{
    bounds::{Bounded, Bounds},
    fc_vec::{FCVec, FCVecContainer, FCVecOps, FCVecRef, FCVecRefMut},
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

union NodeChildren<N, const D: usize, Key, Value> {
    inner: ManuallyDrop<FCVec<Node<N, D, Key, Value>>>,
    leaf: ManuallyDrop<FCVec<(Key, Value)>>,
}

impl<N, const D: usize, Key, Value> NodeChildren<N, D, Key, Value> {
    pub(crate) fn len(&self, level: usize) -> usize {
        if level > 0 {
            unsafe { (*self.inner).len() }
        } else {
            unsafe { (*self.leaf).len() }
        }
    }
}

pub(crate) struct Node<N, const D: usize, Key, Value> {
    pub(crate) bounds: Bounds<N, D>,
    children: NodeChildren<N, D, Key, Value>,
}

impl<N, const D: usize, Key, Value> Node<N, D, Key, Value> {
    unsafe fn new(
        bounds: Bounds<N, D>,
        children: NodeChildren<N, D, Key, Value>,
        _level: usize,
    ) -> Self {
        Node { bounds, children }
    }

    pub(crate) unsafe fn inner_children(&self) -> &FCVec<Node<N, D, Key, Value>> {
        &self.children.inner
    }

    pub(crate) unsafe fn inner_children_mut(&mut self) -> &mut FCVec<Node<N, D, Key, Value>> {
        &mut self.children.inner
    }

    pub(crate) unsafe fn leaf_children(&self) -> &FCVec<(Key, Value)> {
        &self.children.leaf
    }

    pub(crate) unsafe fn leaf_children_mut(&mut self) -> &mut FCVec<(Key, Value)> {
        &mut self.children.leaf
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

pub(crate) struct NodeOps {
    children: FCVecOps,
    min_children: usize,
}

impl NodeOps {
    pub(crate) fn new_ops(min_children: usize, max_children: usize) -> Self {
        NodeOps {
            children: FCVecOps::new_ops(max_children),
            min_children,
        }
    }

    unsafe fn debug_assert_bvh<N, const D: usize, Key, Value>(
        &self,
        node: &Node<N, D, Key, Value>,
        level: usize,
    ) -> Bounds<N, D>
    where
        Key: Bounded<N, D>,
        N: Ord + num_traits::Bounded + Clone + Eq + Debug,
    {
        let children = self.children(node, level);
        let bounds = match children {
            NodeChildrenRef::Inner(children) => {
                Bounds::containing_all(children.iter().map(|child| child.debug_assert_bvh()))
            }
            NodeChildrenRef::Leaf(children) => {
                Bounds::containing_all(children.iter().map(|(key, _)| key.bounds()))
            }
        };

        assert_eq!(node.bounds, bounds);
        bounds
    }

    unsafe fn debug_assert_eq<N, const D: usize, Key, Value>(
        &self,
        a: &Node<N, D, Key, Value>,
        b: &Node<N, D, Key, Value>,
        level: usize,
    ) where
        N: Debug + Eq,
        Key: Debug + Eq,
        Value: Debug + Eq,
    {
        assert_eq!(a.bounds, b.bounds);
        if let Some(level) = NonZeroUsize::new(level) {
            let a_children = self.inner_children(a, level);
            let b_children = self.inner_children(b, level);
            assert_eq!(a_children.len(), b_children.len());
            for (a_child, b_child) in a_children.iter().zip(b_children.iter()) {
                self.debug_assert_eq(a_child.node, b_child.node, level.get() - 1);
            }
        } else {
            let a_children = self.leaf_children(a);
            let b_children = self.leaf_children(b);
            assert_eq!(*a_children, *b_children);
        }
    }

    unsafe fn debug_assert_min_children<N, const D: usize, Key, Value>(
        &self,
        node: &Node<N, D, Key, Value>,
        level: usize,
        is_root: bool,
    ) {
        let children = self.children(node, level);
        if !is_root {
            assert!(children.len() >= self.min_children);
        }
        if let NodeChildrenRef::Inner(children) = children {
            for child in children {
                self.debug_assert_min_children(child.node, level - 1, false);
            }
        }
    }

    pub(crate) fn empty_leaf<'a, N, const D: usize, Key, Value>(
        &'a self,
    ) -> NodeContainer<'a, N, D, Key, Value>
    where
        N: num_traits::Bounded,
    {
        unsafe {
            self.wrap(
                Node::new(
                    Bounds::empty(),
                    NodeChildren {
                        leaf: ManuallyDrop::new(self.children.new().unwrap()),
                    },
                    0,
                ),
                0,
            )
        }
    }

    pub(crate) fn empty_inner<'a, N, const D: usize, Key, Value>(
        &'a self,
        level: NonZeroUsize,
    ) -> NodeContainer<'a, N, D, Key, Value>
    where
        N: num_traits::Bounded,
    {
        unsafe {
            self.wrap(
                Node::new(
                    Bounds::empty(),
                    NodeChildren {
                        inner: ManuallyDrop::new(self.children.new().unwrap()),
                    },
                    level.get(),
                ),
                level.get(),
            )
        }
    }

    unsafe fn take_leaf_children<N, const D: usize, Key, Value>(
        &self,
        node: Node<N, D, Key, Value>,
    ) -> FCVecContainer<(Key, Value)> {
        self.children
            .wrap(ManuallyDrop::into_inner(node.children.leaf))
    }

    unsafe fn take_inner_children<N, const D: usize, Key, Value>(
        &self,
        node: Node<N, D, Key, Value>,
    ) -> FCVecContainer<Node<N, D, Key, Value>> {
        self.children
            .wrap(ManuallyDrop::into_inner(node.children.inner))
    }

    unsafe fn self_insert<'a, 'b, N, const D: usize, Key, Value>(
        &'a self,
        node: &'b mut Node<N, D, Key, Value>,
        level: usize,
        entry: NodeEntry<'a, N, D, Key, Value>,
    ) -> Option<NodeContainer<'a, N, D, Key, Value>>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D>,
    {
        let entry_bounds = entry.bounds();
        match entry {
            NodeEntry::Inner(entry) => {
                let mut children = self.children.wrap_ref_mut(&mut node.children.inner);
                if let Some(overflow_node) = children.try_push(entry.unwrap()) {
                    let (new_bounds, sibling_bounds, sibling_children) =
                        split::quadratic_n(self.min_children, children, overflow_node);
                    node.bounds = new_bounds;
                    Some(self.wrap(
                        Node::new(
                            sibling_bounds,
                            NodeChildren {
                                inner: ManuallyDrop::new(sibling_children.unwrap()),
                            },
                            level,
                        ),
                        level,
                    ))
                } else {
                    node.bounds = Bounds::containing(&node.bounds, &entry_bounds);
                    None
                }
            }
            NodeEntry::Leaf(entry) => {
                let mut children = self.children.wrap_ref_mut(&mut node.children.leaf);
                if let Some(overflow_entry) = children.try_push(entry) {
                    let (new_bounds, sibling_bounds, sibling_children) =
                        split::quadratic_n(self.min_children, children, overflow_entry);
                    node.bounds = new_bounds;
                    Some(self.wrap(
                        Node::new(
                            sibling_bounds,
                            NodeChildren {
                                leaf: ManuallyDrop::new(sibling_children.unwrap()),
                            },
                            level,
                        ),
                        level,
                    ))
                } else {
                    node.bounds = Bounds::containing(&node.bounds, &entry_bounds);
                    None
                }
            }
        }
    }

    unsafe fn insert<'a, 'b, N, const D: usize, Key, Value>(
        &'a self,
        node: &'b mut Node<N, D, Key, Value>,
        level: usize,
        depth: usize,
        entry: NodeEntry<'a, N, D, Key, Value>,
    ) -> Option<NodeContainer<'a, N, D, Key, Value>>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Bounded<N, D>,
    {
        let entry_bounds = entry.bounds();
        if depth > 0 {
            let mut children = self.children.wrap_ref_mut(&mut node.children.inner);
            let mut insert_child = self.wrap_ref_mut(
                select::minimal_volume_increase(children.iter_mut(), &entry.bounds()).unwrap(),
                depth - 1,
            );
            if let Some(new_child) = insert_child.insert(entry) {
                // The child node split, so the entries in new_child are no longer part of self
                // Recompute the bounds of self before trying to insert new_child into self
                node.bounds = Bounds::containing_all(children.iter().map(|child| child.bounds()));
                self.self_insert(node, level, NodeEntry::Inner(new_child))
            } else {
                node.bounds = Bounds::containing(&node.bounds, &entry_bounds);
                None
            }
        } else {
            self.self_insert(node, level, entry)
        }
    }

    unsafe fn remove<N, const D: usize, Key, Value, Q>(
        &self,
        node: &mut Node<N, D, Key, Value>,
        level: usize,
        key: &Q,
        underfull_nodes: &mut [Option<Node<N, D, Key, Value>>],
    ) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq + Borrow<Q>,
        Q: Bounded<N, D> + Eq + ?Sized,
    {
        if level > 0 {
            let mut children = self.children.wrap_ref_mut(&mut node.children.inner);
            let mut i = children.len();
            while i > 0 {
                i -= 1;
                let mut child = self.wrap_ref_mut(&mut children[i], level - 1);
                if child.bounds().intersects(&key.bounds()) {
                    let value = child.remove(key, underfull_nodes);
                    if let Some(value) = value {
                        if child.shallow_len() < self.min_children {
                            drop(child);
                            let child = children.swap_remove(i);
                            underfull_nodes[level - 1] = Some(child);
                        }

                        node.bounds = Bounds::containing_all(
                            node.children.inner.iter().map(|child| child.bounds()),
                        );

                        return Some(value);
                    }
                }
            }
            return None;
        } else {
            let mut children = self.children.wrap_ref_mut(&mut node.children.leaf);
            let index = children.iter().position(|(k, _)| k.borrow() == key);
            if let Some(i) = index {
                let value = children.swap_remove(i).1;
                node.bounds =
                    Bounds::containing_all(node.children.leaf.iter().map(|(key, _)| key.bounds()));

                return Some(value);
            }
            return None;
        }
    }

    pub(crate) unsafe fn drop<N, const D: usize, Key, Value>(
        &self,
        node: &mut Node<N, D, Key, Value>,
        level: usize,
    ) {
        if level > 0 {
            let children = ManuallyDrop::take(&mut node.children.inner);
            let mut children = self.children.wrap(children);
            for child in children.iter_mut() {
                self.drop(child, level - 1);
            }
        } else {
            let children = ManuallyDrop::take(&mut node.children.leaf);
            let _children = self.children.wrap(children);
        }
    }

    unsafe fn clone<'a, N, const D: usize, Key, Value>(
        &'a self,
        node: &Node<N, D, Key, Value>,
        level: usize,
    ) -> NodeContainer<'a, N, D, Key, Value>
    where
        N: Clone,
        Key: Clone,
        Value: Clone,
    {
        let children = self.children(node, level);
        match children {
            NodeChildrenRef::Inner(children) => {
                let mut clone_children = self.children.new();
                for child in children {
                    clone_children.push(child.clone().unwrap());
                }
                self.wrap(
                    Node::new(
                        node.bounds.clone(),
                        NodeChildren {
                            inner: ManuallyDrop::new(clone_children.unwrap()),
                        },
                        level,
                    ),
                    level,
                )
            }
            NodeChildrenRef::Leaf(children) => self.wrap(
                Node::new(
                    node.bounds.clone(),
                    NodeChildren {
                        leaf: ManuallyDrop::new(children.clone().unwrap()),
                    },
                    level,
                ),
                level,
            ),
        }
    }

    unsafe fn len<N, const D: usize, Key, Value>(
        &self,
        node: &Node<N, D, Key, Value>,
        level: usize,
    ) -> usize {
        let children = self.children(node, level);
        match children {
            NodeChildrenRef::Inner(children) => {
                let mut size = 0;
                for child in children {
                    size += child.len();
                }
                size
            }
            NodeChildrenRef::Leaf(children) => children.len(),
        }
    }

    unsafe fn get<'a, 'b, N, const D: usize, Key, Value, Q>(
        &'a self,
        node: &'b Node<N, D, Key, Value>,
        level: usize,
        key: &Q,
    ) -> Option<&'b Value>
    where
        N: Ord,
        Key: Borrow<Q>,
        Q: Bounded<N, D> + Eq + ?Sized,
    {
        let children: NodeChildrenRef<'a, 'b, N, D, Key, Value> = self.children(node, level);
        match children {
            NodeChildrenRef::Inner(children) => {
                for child in children {
                    if child.node.bounds.contains(&key.bounds()) {
                        if let Some(value) = self.get(child.node, level - 1, key) {
                            return Some(value);
                        }
                    }
                }
                None
            }
            NodeChildrenRef::Leaf(children) => children
                .into_iter()
                .find(|(k, _)| k.borrow() == key)
                .map(|(_, v)| v),
        }
    }

    unsafe fn get_mut<'a, N, const D: usize, Key, Value, Q>(
        &self,
        node: &'a mut Node<N, D, Key, Value>,
        level: usize,
        key: &Q,
    ) -> Option<&'a mut Value>
    where
        N: Ord,
        Key: Borrow<Q>,
        Q: Eq + Bounded<N, D> + ?Sized,
    {
        if level > 0 {
            let children = self.children.wrap_ref_mut(&mut node.children.inner);
            for child in children {
                if child.bounds.contains(&key.bounds()) {
                    if let Some(value) = self.get_mut(child, level - 1, key) {
                        return Some(value);
                    }
                }
            }
            None
        } else {
            let children = self.children.wrap_ref_mut(&mut node.children.leaf);
            children
                .iter_mut_take()
                .find(|(k, _)| k.borrow() == key)
                .map(|(_, v)| v)
        }
    }

    unsafe fn insert_unique<'a, 'b, N, const D: usize, Key, Value>(
        &'a self,
        node: &'b mut Node<N, D, Key, Value>,
        level: usize,
        key: Key,
        value: Value,
    ) -> (Option<Value>, Option<NodeContainer<'a, N, D, Key, Value>>)
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Eq + Bounded<N, D>,
    {
        let entry_bounds = key.bounds();
        if level > 0 {
            let mut children = self.children.wrap_ref_mut(&mut node.children.inner);
            // Check which children contain the key bounds
            let children_containing_key = children
                .iter_mut()
                .filter(|child| child.bounds.contains(&entry_bounds));
            match get_only(children_containing_key) {
                GetOnlyResult::None => {
                    // If there are no children containing the key bounds, then
                    // the key cannot exist in the node.
                    (
                        None,
                        self.insert(node, level, level, NodeEntry::Leaf((key, value))),
                    )
                }
                GetOnlyResult::Only(child) => {
                    let mut child = self.wrap_ref_mut(child, level - 1);
                    // If there is only one children containing the key bounds,
                    // then we can maintain the uniqueness invariant by
                    // performing a unique insert into that child.
                    let (old_value, new_child) = child.insert_unique(key, value);

                    (
                        old_value,
                        if let Some(new_child) = new_child {
                            // The child node split, so the entries in new_child are no longer part of self
                            // Recompute the bounds of self before trying to insert new_child into self
                            node.bounds =
                                Bounds::containing_all(children.iter().map(|child| child.bounds()));
                            self.self_insert(node, level, NodeEntry::Inner(new_child))
                        } else {
                            node.bounds = Bounds::containing(&node.bounds, &entry_bounds);
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
                            self.insert(node, level, level, NodeEntry::Leaf((key, value))),
                        )
                    }
                }
            }
        } else {
            let mut children = self.children.wrap_ref_mut(&mut node.children.leaf);
            // Try to find the key among the children and overwrite it if it
            // exists. If it doesn't exist, perform a regular insert.
            if let Some(entry) = children.iter_mut().find(|(k, _)| k == &key) {
                (Some(std::mem::replace(&mut entry.1, value)), None)
            } else {
                (
                    None,
                    self.insert(node, level, level, NodeEntry::Leaf((key, value))),
                )
            }
        }
    }

    /// Branches the root node into two nodes in-place, such that the previous
    /// root node and the sibling node become children of the new root node.
    unsafe fn root_branch<'a, N, const D: usize, Key, Value>(
        &'a self,
        root: &mut Node<N, D, Key, Value>,
        height: &mut usize,
        sibling: NodeContainer<'a, N, D, Key, Value>,
    ) where
        N: Ord + Clone + num_traits::Bounded,
    {
        let bounds = Bounds::containing(&root.bounds, &sibling.node.bounds);
        let mut next_root_children = self.children.new();
        next_root_children.push(ptr::read(root));
        next_root_children.push(sibling.unwrap());
        ptr::write(
            root,
            Node::new(
                bounds,
                NodeChildren {
                    inner: ManuallyDrop::new(next_root_children.unwrap()),
                },
                *height,
            ),
        );
        *height += 1;
    }

    unsafe fn root_insert_entry<N, const D: usize, Key, Value>(
        &self,
        root: &mut Node<N, D, Key, Value>,
        height: &mut usize,
        depth: usize, // TODO: Get rid of this argument
        entry: NodeEntry<N, D, Key, Value>,
    ) where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Bounded<N, D>,
    {
        if let Some(sibling) = self.insert(root, *height, *height - depth, entry) {
            self.root_branch(root, height, sibling);
        }
    }

    unsafe fn root_insert<N, const D: usize, Key, Value>(
        &self,
        root: &mut Node<N, D, Key, Value>,
        height: &mut usize,
        key: Key,
        value: Value,
    ) where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Bounded<N, D>,
    {
        self.root_insert_entry(root, height, 0, NodeEntry::Leaf((key, value)));
    }

    unsafe fn root_insert_unique<N, const D: usize, Key, Value>(
        &self,
        root: &mut Node<N, D, Key, Value>,
        height: &mut usize,
        key: Key,
        value: Value,
    ) -> Option<Value>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Eq + Bounded<N, D>,
    {
        let (prev_value, sibling) = self.insert_unique(root, *height, key, value);
        if let Some(sibling) = sibling {
            self.root_branch(root, height, sibling);
        }
        prev_value
    }

    /// Tries to unbranch the root node in-place, such that the root node becomes
    /// the only child of the previous root node.
    unsafe fn root_try_unbranch<N, const D: usize, Key, Value>(
        &self,
        root: &mut Node<N, D, Key, Value>,
        level: &mut usize,
    ) where
        N: num_traits::Bounded,
    {
        if *level > 0 {
            let mut children = self.children.wrap_ref_mut(&mut root.children.inner);
            if children.len() == 1 {
                let new_root = children.remove(0);
                unsafe {
                    self.drop(root, *level);
                }
                *root = new_root;
                *level -= 1;
            }
        }
    }

    unsafe fn root_remove<N, const D: usize, Key, Value, Q>(
        &self,
        root: &mut Node<N, D, Key, Value>,
        height: &mut usize,
        key: &Q,
    ) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Borrow<Q> + Bounded<N, D> + Eq,
        Q: Bounded<N, D> + Eq + ?Sized,
    {
        let original_height = *height;
        let mut underfull_nodes: Box<[Option<Node<N, D, Key, Value>>]> =
            std::iter::repeat_with(|| None).take(*height).collect();
        if let Some(value) = self.remove(root, *height, key, &mut underfull_nodes) {
            self.root_try_unbranch(root, height);

            // reinsert entries at leaf level
            if original_height > 0 {
                if let Some(undefull_leaf) = underfull_nodes[0].take() {
                    let children = unsafe { self.wrap(undefull_leaf, 0).take_leaf_children() };
                    for leaf_entry in children {
                        unsafe {
                            self.root_insert_entry(root, height, 0, NodeEntry::Leaf(leaf_entry));
                        }
                    }
                }
            }

            // reinsert entries at inner levels
            for level in 1..original_height {
                if let Some(children) = underfull_nodes[level].take() {
                    let children = unsafe { self.wrap(children, level).take_inner_children() };
                    for node in children {
                        unsafe {
                            self.root_insert_entry(
                                root,
                                height,
                                level,
                                NodeEntry::Inner(self.wrap(node, level - 1)),
                            );
                        }
                    }
                }
            }

            return Some(value);
        }
        return None;
    }

    unsafe fn children<'a, 'b, N, const D: usize, Key, Value>(
        &'a self,
        node: &'b Node<N, D, Key, Value>,
        level: usize,
    ) -> NodeChildrenRef<'a, 'b, N, D, Key, Value> {
        if let Some(level) = NonZeroUsize::new(level) {
            NodeChildrenRef::Inner(InnerNodeChildrenRef {
                ops: self,
                level,
                children: &node.children.inner,
            })
        } else {
            NodeChildrenRef::Leaf(self.children.wrap_ref(&node.children.leaf))
        }
    }

    unsafe fn inner_children<'a, 'b, N, const D: usize, Key, Value>(
        &'a self,
        node: &'b Node<N, D, Key, Value>,
        level: NonZeroUsize,
    ) -> InnerNodeChildrenRef<'a, 'b, N, D, Key, Value> {
        InnerNodeChildrenRef {
            ops: self,
            level,
            children: &node.children.inner,
        }
    }

    unsafe fn leaf_children<'a, 'b, N, const D: usize, Key, Value>(
        &'a self,
        node: &'b Node<N, D, Key, Value>,
    ) -> FCVecRef<'a, 'b, (Key, Value)> {
        self.children.wrap_ref(&node.children.leaf)
    }

    unsafe fn children_mut<'a, 'b, N, const D: usize, Key, Value>(
        &'a self,
        node: &'b mut Node<N, D, Key, Value>,
        level: usize,
    ) -> NodeChildrenRefMut<'a, 'b, N, D, Key, Value> {
        if let Some(level) = NonZeroUsize::new(level) {
            NodeChildrenRefMut::Inner(InnerNodeChildrenRefMut {
                ops: self,
                level,
                children: &mut node.children.inner,
            })
        } else {
            NodeChildrenRefMut::Leaf(self.children.wrap_ref_mut(&mut node.children.leaf))
        }
    }

    pub(crate) unsafe fn wrap<N, const D: usize, Key, Value>(
        &self,
        node: Node<N, D, Key, Value>,
        level: usize,
    ) -> NodeContainer<N, D, Key, Value> {
        NodeContainer {
            ops: self,
            level,
            node,
        }
    }

    pub(crate) unsafe fn wrap_ref<'a, 'b, N, const D: usize, Key, Value>(
        &'a self,
        node: &'b Node<N, D, Key, Value>,
        level: usize,
    ) -> NodeRef<'a, 'b, N, D, Key, Value> {
        NodeRef {
            ops: self,
            level,
            node: node,
        }
    }

    pub(crate) unsafe fn wrap_ref_mut<'a, 'b, N, const D: usize, Key, Value>(
        &'a self,
        node: &'b mut Node<N, D, Key, Value>,
        level: usize,
    ) -> NodeRefMut<'a, 'b, N, D, Key, Value> {
        NodeRefMut {
            ops: self,
            level,
            node: node,
        }
    }

    pub(crate) unsafe fn wrap_root_ref_mut<'a, 'b, N, const D: usize, Key, Value>(
        &'a self,
        node: &'b mut Node<N, D, Key, Value>,
        height: &'b mut usize,
    ) -> RootNodeRefMut<'a, 'b, N, D, Key, Value> {
        RootNodeRefMut {
            ops: self,
            height,
            node: node,
        }
    }
}

pub(crate) struct NodeRefMut<'a, 'b, N, const D: usize, Key, Value> {
    ops: &'a NodeOps,
    level: usize,
    node: &'b mut Node<N, D, Key, Value>,
}

impl<'a, 'b, N, const D: usize, Key, Value> NodeRefMut<'a, 'b, N, D, Key, Value> {
    unsafe fn insert(
        &mut self,
        entry: NodeEntry<'a, N, D, Key, Value>,
    ) -> Option<NodeContainer<'a, N, D, Key, Value>>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Bounded<N, D>,
    {
        unsafe {
            self.ops
                .insert(&mut self.node, self.level, self.level, entry)
        }
    }

    fn remove<Q>(
        &mut self,
        key: &Q,
        underfull_nodes: &mut [Option<Node<N, D, Key, Value>>],
    ) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq + Borrow<Q>,
        Q: Bounded<N, D> + Eq + ?Sized,
    {
        unsafe {
            self.ops
                .remove(&mut self.node, self.level, key, underfull_nodes)
        }
    }

    pub(crate) fn get_mut<Q>(self, key: &Q) -> Option<&'b mut Value>
    where
        N: Ord,
        Key: Borrow<Q>,
        Q: Eq + Bounded<N, D> + ?Sized,
    {
        unsafe { self.ops.get_mut(self.node, self.level, key) }
    }

    pub(crate) fn insert_unique(
        &mut self,
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
                    .insert_unique(&mut self.node, self.level, key, value);
            (prev_value, new_sibling)
        }
    }

    fn bounds(&self) -> &Bounds<N, D> {
        &self.node.bounds
    }

    fn shallow_len(&self) -> usize {
        self.node.children.len(self.level)
    }
}

pub(crate) struct RootNodeRefMut<'a, 'b, N, const D: usize, Key, Value> {
    ops: &'a NodeOps,
    height: &'b mut usize,
    node: &'b mut Node<N, D, Key, Value>,
}

impl<'a, 'b, N, const D: usize, Key, Value> RootNodeRefMut<'a, 'b, N, D, Key, Value> {
    pub(crate) fn insert(&mut self, key: Key, value: Value)
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Bounded<N, D>,
    {
        unsafe { self.ops.root_insert(self.node, self.height, key, value) }
    }

    pub(crate) fn insert_unique(&mut self, key: Key, value: Value) -> Option<Value>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Eq + Bounded<N, D>,
    {
        unsafe {
            self.ops
                .root_insert_unique(self.node, self.height, key, value)
        }
    }

    pub(crate) fn remove<Q>(&mut self, key: &Q) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq + Borrow<Q>,
        Q: Bounded<N, D> + Eq + ?Sized,
    {
        unsafe { self.ops.root_remove(self.node, self.height, key) }
    }
}

pub(crate) struct NodeContainer<'a, N, const D: usize, Key, Value> {
    ops: &'a NodeOps,
    level: usize,
    node: Node<N, D, Key, Value>,
}

impl<'a, N, const D: usize, Key, Value> NodeContainer<'a, N, D, Key, Value> {
    fn children(&'a self) -> NodeChildrenRef<'a, 'a, N, D, Key, Value> {
        unsafe { self.ops.children(&self.node, self.level) }
    }

    pub(crate) unsafe fn take_leaf_children(self) -> FCVecContainer<'a, (Key, Value)> {
        if self.level > 0 {
            panic!("Cannot take leaf children from an inner node");
        }
        let ops = ptr::read(&self.ops);
        let node = ptr::read(&self.node);
        mem::forget(self);
        unsafe { ops.take_leaf_children(node) }
    }

    pub(crate) unsafe fn take_inner_children(self) -> FCVecContainer<'a, Node<N, D, Key, Value>> {
        if self.level == 0 {
            panic!("Cannot take inner children from a leaf node");
        }
        let ops = ptr::read(&self.ops);
        let node = ptr::read(&self.node);
        mem::forget(self);
        unsafe { ops.take_inner_children(node) }
    }

    /// Unwraps the node from the container without dropping it.
    pub(crate) unsafe fn unwrap(self) -> Node<N, D, Key, Value> {
        let s = ManuallyDrop::new(self);
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
        unsafe { self.ops.clone(&self.node, self.level) }
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
            .field("node", &self.children())
            .finish()
    }
}

pub(crate) struct NodeRef<'a, 'b, N, const D: usize, Key, Value> {
    ops: &'a NodeOps,
    level: usize,
    node: &'b Node<N, D, Key, Value>,
}

impl<'a, 'b, N, const D: usize, Key, Value> NodeRef<'a, 'b, N, D, Key, Value> {
    pub(crate) fn children(&self) -> NodeChildrenRef<'a, 'b, N, D, Key, Value> {
        unsafe { self.ops.children(&self.node, self.level) }
    }

    pub(crate) fn len(&self) -> usize {
        unsafe { self.ops.len(self.node, self.level) }
    }

    pub(crate) fn get<Q>(&self, key: &Q) -> Option<&'b Value>
    where
        N: Ord,
        Key: Borrow<Q>,
        Q: Eq + Bounded<N, D> + ?Sized,
    {
        unsafe { self.ops.get(self.node, self.level, key) }
    }

    pub(crate) fn clone(&self) -> NodeContainer<'a, N, D, Key, Value>
    where
        N: Clone,
        Key: Clone,
        Value: Clone,
    {
        unsafe { self.ops.clone(self.node, self.level) }
    }

    pub(crate) fn debug_assert_bvh(&self) -> Bounds<N, D>
    where
        Key: Bounded<N, D>,
        N: Ord + num_traits::Bounded + Clone + Eq + Debug,
    {
        unsafe { self.ops.debug_assert_bvh(self.node, self.level) }
    }

    pub(crate) fn debug_assert_eq(&self, other: NodeRef<N, D, Key, Value>)
    where
        N: Debug + Eq,
        Key: Debug + Eq,
        Value: Debug + Eq,
    {
        unsafe { self.ops.debug_assert_eq(self.node, other.node, self.level) }
    }

    pub(crate) fn debug_assert_min_children(&self, is_root: bool) {
        unsafe {
            self.ops
                .debug_assert_min_children(self.node, self.level, is_root)
        }
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
            .field("node", &self.children())
            .finish()
    }
}

pub(crate) struct InnerNodeChildrenRef<'a, 'b, N, const D: usize, Key, Value> {
    ops: &'a NodeOps,
    level: NonZeroUsize,
    children: &'b FCVec<Node<N, D, Key, Value>>,
}

impl<'a, 'b, N, const D: usize, Key, Value> Debug for InnerNodeChildrenRef<'a, 'b, N, D, Key, Value>
where
    N: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, 'b, N, const D: usize, Key, Value> InnerNodeChildrenRef<'a, 'b, N, D, Key, Value> {
    fn len(&self) -> usize {
        self.children.len()
    }

    fn iter(&self) -> InnerNodeChildrenRefIter<'a, 'b, N, D, Key, Value> {
        InnerNodeChildrenRefIter {
            ops: self.ops,
            level: self.level,
            children: self.children.iter(),
        }
    }
}

pub(crate) struct InnerNodeChildrenRefIter<'a, 'b, N, const D: usize, Key, Value> {
    ops: &'a NodeOps,
    level: NonZeroUsize,
    children: slice::Iter<'b, Node<N, D, Key, Value>>,
}

impl<'a, 'b, N, const D: usize, Key, Value> IntoIterator
    for InnerNodeChildrenRef<'a, 'b, N, D, Key, Value>
{
    type Item = NodeRef<'a, 'b, N, D, Key, Value>;
    type IntoIter = InnerNodeChildrenRefIter<'a, 'b, N, D, Key, Value>;

    fn into_iter(self) -> Self::IntoIter {
        InnerNodeChildrenRefIter {
            ops: self.ops,
            level: self.level,
            children: self.children.iter(),
        }
    }
}

impl<'a, 'b, N, const D: usize, Key, Value> Iterator
    for InnerNodeChildrenRefIter<'a, 'b, N, D, Key, Value>
{
    type Item = NodeRef<'a, 'b, N, D, Key, Value>;

    fn next(&mut self) -> Option<Self::Item> {
        self.children
            .next()
            .map(|node| unsafe { self.ops.wrap_ref(node, self.level.get() - 1) })
    }
}

pub(crate) enum NodeChildrenRef<'a, 'b, N, const D: usize, Key, Value> {
    Inner(InnerNodeChildrenRef<'a, 'b, N, D, Key, Value>),
    Leaf(FCVecRef<'a, 'b, (Key, Value)>),
}

impl<'a, 'b, N, const D: usize, Key, Value> Debug for NodeChildrenRef<'a, 'b, N, D, Key, Value>
where
    N: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeChildrenRef::Inner(children) => children.fmt(f),
            NodeChildrenRef::Leaf(children) => children.fmt(f),
        }
    }
}

impl<'a, 'b, N, const D: usize, Key, Value> NodeChildrenRef<'a, 'b, N, D, Key, Value> {
    fn len(&self) -> usize {
        match self {
            NodeChildrenRef::Inner(children) => children.len(),
            NodeChildrenRef::Leaf(children) => children.len(),
        }
    }
}

struct InnerNodeChildrenRefMut<'a, 'b, N, const D: usize, Key, Value> {
    ops: &'a NodeOps,
    level: NonZeroUsize,
    children: &'b mut FCVec<Node<N, D, Key, Value>>,
}

enum NodeChildrenRefMut<'a, 'b, N, const D: usize, Key, Value> {
    Inner(InnerNodeChildrenRefMut<'a, 'b, N, D, Key, Value>),
    Leaf(FCVecRefMut<'a, 'b, (Key, Value)>),
}
