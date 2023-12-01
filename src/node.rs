use std::{
    fmt::Debug,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    num::NonZeroUsize,
    ops::Sub,
    ptr,
};

use crate::{
    bounds::{empty_bounds, min_bounds, min_bounds_all, Bounded, Bounds},
    fc_vec::{FCVec, FCVecContainer, FCVecOps},
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

    _phantom: PhantomData<(Key, Value)>,
}

impl<N, const D: usize, Key, Value> Node<N, D, Key, Value> {
    unsafe fn new(
        bounds: Bounds<N, D>,
        children: NodeChildren<N, D, Key, Value>,
        _level: usize,
    ) -> Self {
        Node {
            bounds,
            children,

            _phantom: PhantomData,
        }
    }

    pub(crate) unsafe fn inner_children(&self) -> &FCVec<Node<N, D, Key, Value>> {
        &self.children.inner
    }

    pub(crate) unsafe fn leaf_children(&self) -> &FCVec<(Key, Value)> {
        &self.children.leaf
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

pub(crate) struct NodeOps<N, const D: usize, Key, Value> {
    leaf: FCVecOps<(Key, Value)>,
    inner: FCVecOps<Node<N, D, Key, Value>>,
    min_children: usize,
}

impl<N, const D: usize, Key, Value> NodeOps<N, D, Key, Value> {
    pub(crate) fn new_ops(min_children: usize, max_children: usize) -> Self {
        NodeOps {
            leaf: FCVecOps::new_ops(max_children),
            inner: FCVecOps::new_ops(max_children),
            min_children,
        }
    }

    pub(crate) unsafe fn debug_assert_bvh(
        &self,
        node: &Node<N, D, Key, Value>,
        level: usize,
    ) -> Bounds<N, D>
    where
        Key: Bounded<N, D>,
        N: Ord + num_traits::Bounded + Clone + Eq + Debug,
    {
        let bounds = if level > 0 {
            let children = self.inner.wrap_ref(&node.children.inner);
            min_bounds_all(
                children
                    .iter()
                    .map(|node| self.debug_assert_bvh(node, level - 1)),
            )
        } else {
            let children = self.leaf.wrap_ref(&node.children.leaf);
            min_bounds_all(children.iter().map(|(key, _)| key.bounds()))
        };
        assert_eq!(node.bounds, bounds);
        bounds
    }

    pub(crate) unsafe fn debug_assert_eq(
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
        if level > 0 {
            let a_children = self.inner.wrap_ref(&a.children.inner);
            let b_children = self.inner.wrap_ref(&b.children.inner);
            assert_eq!(a_children.len(), b_children.len());
            for (a_child, b_child) in a_children.iter().zip(b_children.iter()) {
                self.debug_assert_eq(a_child, b_child, level - 1);
            }
        } else {
            let a_children = self.leaf.wrap_ref(&a.children.leaf);
            let b_children = self.leaf.wrap_ref(&b.children.leaf);
            assert_eq!(*a_children, *b_children);
        }
    }

    pub(crate) unsafe fn debug_assert_min_children(
        &self,
        node: &Node<N, D, Key, Value>,
        level: usize,
        is_root: bool,
    ) {
        if !is_root {
            assert!(
                if level > 0 {
                    let children = self.inner.wrap_ref(&node.children.inner);
                    children.len()
                } else {
                    let children = self.leaf.wrap_ref(&node.children.leaf);
                    children.len()
                } >= self.min_children
            );
        }
        if level > 0 {
            let children = self.inner.wrap_ref(&node.children.inner);
            for child in children {
                self.debug_assert_min_children(child, level - 1, false);
            }
        }
    }

    pub(crate) fn empty_leaf<'a>(&'a self) -> NodeContainer<'a, N, D, Key, Value>
    where
        N: num_traits::Bounded,
    {
        unsafe {
            self.wrap(
                Node::new(
                    empty_bounds(),
                    NodeChildren {
                        leaf: ManuallyDrop::new(self.leaf.new().unwrap()),
                    },
                    0,
                ),
                0,
            )
        }
    }

    pub(crate) unsafe fn empty_inner<'a>(
        &'a self,
        level: NonZeroUsize,
    ) -> NodeContainer<'a, N, D, Key, Value>
    where
        N: num_traits::Bounded,
    {
        unsafe {
            self.wrap(
                Node::new(
                    empty_bounds(),
                    NodeChildren {
                        inner: ManuallyDrop::new(self.inner.new().unwrap()),
                    },
                    level.get(),
                ),
                level.get(),
            )
        }
    }

    unsafe fn take_leaf_children(
        &self,
        node: Node<N, D, Key, Value>,
    ) -> FCVecContainer<(Key, Value)> {
        self.leaf.wrap(ManuallyDrop::into_inner(node.children.leaf))
    }

    unsafe fn take_inner_children(
        &self,
        node: Node<N, D, Key, Value>,
    ) -> FCVecContainer<Node<N, D, Key, Value>> {
        self.inner
            .wrap(ManuallyDrop::into_inner(node.children.inner))
    }

    unsafe fn self_insert<'a, 'b>(
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
                let mut children = self.inner.wrap_ref_mut(&mut node.children.inner);
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
                    node.bounds = min_bounds(&node.bounds, &entry_bounds);
                    None
                }
            }
            NodeEntry::Leaf(entry) => {
                let mut children = self.leaf.wrap_ref_mut(&mut node.children.leaf);
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
                    node.bounds = min_bounds(&node.bounds, &entry_bounds);
                    None
                }
            }
        }
    }

    pub(crate) unsafe fn insert<'a, 'b>(
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
            let mut children = self.inner.wrap_ref_mut(&mut node.children.inner);
            let mut insert_child = self.wrap_ref_mut(
                select::minimal_volume_increase(children.iter_mut(), &entry.bounds()).unwrap(),
                depth - 1,
            );
            if let Some(new_child) = insert_child.insert(entry) {
                // The child node split, so the entries in new_child are no longer part of self
                // Recompute the bounds of self before trying to insert new_child into self
                node.bounds = min_bounds_all(children.iter().map(|child| child.bounds()));
                self.self_insert(node, level, NodeEntry::Inner(new_child))
            } else {
                node.bounds = min_bounds(&node.bounds, &entry_bounds);
                None
            }
        } else {
            self.self_insert(node, level, entry)
        }
    }

    pub(crate) unsafe fn remove(
        &self,
        node: &mut Node<N, D, Key, Value>,
        level: usize,
        key: &Key,
        underfull_nodes: &mut [Option<Node<N, D, Key, Value>>],
    ) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq,
    {
        if level > 0 {
            let mut children = self.inner.wrap_ref_mut(&mut node.children.inner);
            let mut i = children.len();
            while i > 0 {
                i -= 1;
                if children[i].bounds.intersects(&key.bounds()) {
                    if let Some(value) =
                        self.remove(&mut children[i], level - 1, key, underfull_nodes)
                    {
                        if children[i].children.len(level - 1) < self.min_children {
                            let removed_child = children.swap_remove(i);
                            underfull_nodes[level - 1] = Some(removed_child);
                        }

                        node.bounds =
                            min_bounds_all(node.children.inner.iter().map(|child| child.bounds()));

                        return Some(value);
                    }
                }
            }
            return None;
        } else {
            let mut children = self.leaf.wrap_ref_mut(&mut node.children.leaf);
            let index = children.iter().position(|(k, _)| k == key);
            if let Some(i) = index {
                let value = children.swap_remove(i).1;
                node.bounds =
                    min_bounds_all(node.children.leaf.iter().map(|(key, _)| key.bounds()));

                return Some(value);
            }
            return None;
        }
    }

    pub(crate) unsafe fn drop(&self, node: &mut Node<N, D, Key, Value>, level: usize) {
        if level > 0 {
            let children = ManuallyDrop::take(&mut node.children.inner);
            let mut children = self.inner.wrap(children);
            for child in children.iter_mut() {
                self.drop(child, level - 1);
            }
        } else {
            let children = ManuallyDrop::take(&mut node.children.leaf);
            let _children = self.leaf.wrap(children);
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
            let children = self.inner.wrap_ref(&node.children.inner);
            let mut clone_children = self.inner.new();
            for child in children {
                clone_children.push(self.clone(child, level - 1));
            }
            Node::new(
                node.bounds.clone(),
                NodeChildren {
                    inner: ManuallyDrop::new(clone_children.unwrap()),
                },
                level,
            )
        } else {
            let children = self.leaf.wrap_ref(&node.children.leaf);
            Node::new(
                node.bounds.clone(),
                NodeChildren {
                    leaf: ManuallyDrop::new(children.clone().unwrap()),
                },
                level,
            )
        }
    }

    pub(crate) unsafe fn len(&self, node: &Node<N, D, Key, Value>, level: usize) -> usize {
        if level > 0 {
            let children = self.inner.wrap_ref(&node.children.inner);
            let mut size = 0;
            for child in children {
                size += self.len(child, level - 1);
            }
            size
        } else {
            let children = self.leaf.wrap_ref(&node.children.leaf);
            children.len()
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
            let children = self.inner.wrap_ref(&node.children.inner);
            for child in children {
                if child.bounds.contains(&key.bounds()) {
                    if let Some(value) = self.get(child, level - 1, key) {
                        return Some(value);
                    }
                }
            }
            None
        } else {
            node.children
                .leaf
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
            let children = self.inner.wrap_ref_mut(&mut node.children.inner);
            for child in children {
                if child.bounds.contains(&key.bounds()) {
                    if let Some(value) = self.get_mut(child, level - 1, key) {
                        return Some(value);
                    }
                }
            }
            None
        } else {
            let children = self.leaf.wrap_ref_mut(&mut node.children.leaf);
            children
                .iter_mut_take()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
        }
    }

    pub(crate) unsafe fn insert_unique<'a, 'b>(
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
            let mut children = self.inner.wrap_ref_mut(&mut node.children.inner);
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
                                min_bounds_all(children.iter().map(|child| child.bounds()));
                            self.self_insert(node, level, NodeEntry::Inner(new_child))
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
                            self.insert(node, level, level, NodeEntry::Leaf((key, value))),
                        )
                    }
                }
            }
        } else {
            let mut children = self.leaf.wrap_ref_mut(&mut node.children.leaf);
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
    unsafe fn root_branch<'a>(
        &'a self,
        root: &mut Node<N, D, Key, Value>,
        height: &mut usize,
        sibling: NodeContainer<'a, N, D, Key, Value>,
    ) where
        N: Ord + Clone + num_traits::Bounded,
    {
        let bounds = min_bounds(&root.bounds, &sibling.node.bounds);
        let mut next_root_children = self.inner.new();
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

    unsafe fn root_insert_entry(
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

    unsafe fn root_insert(
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

    unsafe fn root_insert_unique(
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
    unsafe fn root_try_unbranch(&self, root: &mut Node<N, D, Key, Value>, level: &mut usize)
    where
        N: num_traits::Bounded,
    {
        if *level > 0 {
            let mut children = self.inner.wrap_ref_mut(&mut root.children.inner);
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

    unsafe fn root_remove(
        &self,
        root: &mut Node<N, D, Key, Value>,
        height: &mut usize,
        key: &Key,
    ) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq,
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

    pub(crate) unsafe fn wrap(
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

    pub(crate) unsafe fn wrap_ref<'a, 'b>(
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

    pub(crate) unsafe fn wrap_ref_mut<'a, 'b>(
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

    pub(crate) unsafe fn wrap_root_ref_mut<'a, 'b>(
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
    ops: &'a NodeOps<N, D, Key, Value>,
    level: usize,
    node: &'b mut Node<N, D, Key, Value>,
}

impl<'a, 'b, N, const D: usize, Key, Value> NodeRefMut<'a, 'b, N, D, Key, Value> {
    pub(crate) unsafe fn insert(
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

    fn remove(
        &mut self,
        key: &Key,
        underfull_nodes: &mut [Option<Node<N, D, Key, Value>>],
    ) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq,
        Value: Eq,
    {
        unsafe {
            self.ops
                .remove(&mut self.node, self.level, key, underfull_nodes)
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
}

pub(crate) struct RootNodeRefMut<'a, 'b, N, const D: usize, Key, Value> {
    ops: &'a NodeOps<N, D, Key, Value>,
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

    pub(crate) fn remove(&mut self, key: &Key) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq,
    {
        unsafe { self.ops.root_remove(self.node, self.height, key) }
    }
}

pub(crate) struct NodeContainer<'a, N, const D: usize, Key, Value> {
    ops: &'a NodeOps<N, D, Key, Value>,
    level: usize,
    node: Node<N, D, Key, Value>,
}

impl<'a, N, const D: usize, Key, Value> NodeContainer<'a, N, D, Key, Value> {
    fn insert(
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

    fn remove(
        &mut self,
        key: &Key,
        underfully_nodes: &mut [Option<Node<N, D, Key, Value>>],
    ) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq,
        Value: Eq,
    {
        unsafe {
            self.ops
                .remove(&mut self.node, self.level, key, underfully_nodes)
        }
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
    children: &'b NodeChildren<N, D, Key, Value>,

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
                    self.children.inner.iter().map(|node| NodeRef {
                        ops: self.ops,
                        level: self.level - 1,
                        node,
                    })
                })
                .finish()
        } else {
            f.debug_list()
                .entries(unsafe { self.children.leaf.iter().map(|(key, value)| (key, value)) })
                .finish()
        }
    }
}
