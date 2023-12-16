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
    contains::Contains,
    fc_vec::{self, FCVec, FCVecContainer, FCVecOps, FCVecRef, FCVecRefMut},
    intersects::Intersects,
    select, split,
    util::{get_only, GetOnlyResult},
};

pub(crate) enum NodeEntry<N, const D: usize, Key, Value> {
    Inner(NodeContainer<N, D, Key, Value>),
    Leaf((Key, Value)),
}

impl<N, const D: usize, Key, Value> NodeEntry<N, D, Key, Value> {
    fn target_level(&self) -> usize {
        match self {
            NodeEntry::Inner(node) => node.level + 1,
            NodeEntry::Leaf(_) => 0,
        }
    }
}

impl<N, const D: usize, Key, Value> Bounded<N, D> for NodeEntry<N, D, Key, Value>
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

#[derive(Copy, Clone)]
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

    pub(crate) fn empty_leaf<N, const D: usize, Key, Value>(
        &self,
    ) -> NodeContainer<N, D, Key, Value>
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

    unsafe fn wrap<N, const D: usize, Key, Value>(
        &self,
        node: Node<N, D, Key, Value>,
        level: usize,
    ) -> NodeContainer<N, D, Key, Value> {
        NodeContainer {
            ops: *self,
            level,
            node,
        }
    }

    pub(crate) unsafe fn wrap_ref<'a, N, const D: usize, Key, Value>(
        &self,
        node: &'a Node<N, D, Key, Value>,
        level: usize,
    ) -> NodeRef<'a, N, D, Key, Value> {
        NodeRef {
            ops: *self,
            level,
            node: node,
        }
    }

    pub(crate) unsafe fn wrap_ref_mut<'a, N, const D: usize, Key, Value>(
        &self,
        node: &'a mut Node<N, D, Key, Value>,
        level: usize,
    ) -> NodeRefMut<'a, N, D, Key, Value> {
        NodeRefMut {
            ops: *self,
            level,
            node: node,
        }
    }

    pub(crate) unsafe fn wrap_root_ref_mut<'a, N, const D: usize, Key, Value>(
        &self,
        node: &'a mut Node<N, D, Key, Value>,
        height: &'a mut usize,
    ) -> RootNodeRefMut<'a, N, D, Key, Value> {
        RootNodeRefMut {
            ops: *self,
            height,
            node: node,
        }
    }

    unsafe fn wrap_children<N, const D: usize, Key, Value>(
        &self,
        children: NodeChildren<N, D, Key, Value>,
        level: usize,
    ) -> NodeChildrenContainer<N, D, Key, Value> {
        if let Some(level) = NonZeroUsize::new(level) {
            NodeChildrenContainer::Inner(InnerNodeChildrenContainer {
                ops: *self,
                level,
                children: ManuallyDrop::into_inner(children.inner),
            })
        } else {
            NodeChildrenContainer::Leaf(self.children.wrap(ManuallyDrop::into_inner(children.leaf)))
        }
    }
}

pub(crate) struct NodeRefMut<'a, N, const D: usize, Key, Value> {
    ops: NodeOps,
    level: usize,
    node: &'a mut Node<N, D, Key, Value>,
}

impl<'a, N, const D: usize, Key, Value> NodeRefMut<'a, N, D, Key, Value> {
    fn self_insert(
        &mut self,
        entry: NodeEntry<N, D, Key, Value>,
    ) -> Option<NodeContainer<N, D, Key, Value>>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D>,
    {
        let entry_bounds = entry.bounds();
        let min_children = self.ops.min_children;
        match (self.children_mut(), entry) {
            (NodeChildrenRefMut::Inner(mut children), NodeEntry::Inner(entry)) => {
                if children.level.get() - 1 != entry.level {
                    panic!(
                        "cannot insert entry with level {} in node with level {}",
                        entry.level, children.level
                    );
                }
                if let Some(overflow_node) = children.try_push(entry) {
                    let (new_bounds, sibling) = children.split(overflow_node);
                    self.node.bounds = new_bounds;
                    Some(sibling)
                } else {
                    self.node.bounds = Bounds::union(&self.node.bounds, &entry_bounds);
                    None
                }
            }
            (NodeChildrenRefMut::Leaf(mut children), NodeEntry::Leaf(entry)) => {
                if let Some(overflow_entry) = children.try_push(entry) {
                    let (new_bounds, sibling_bounds, sibling_children) =
                        split::quadratic(min_children, children, overflow_entry);
                    self.node.bounds = new_bounds;
                    Some(unsafe {
                        self.ops.wrap(
                            Node::new(
                                sibling_bounds,
                                NodeChildren {
                                    leaf: ManuallyDrop::new(sibling_children.unwrap()),
                                },
                                self.level,
                            ),
                            self.level,
                        )
                    })
                } else {
                    self.node.bounds = Bounds::union(&self.node.bounds, &entry_bounds);
                    None
                }
            }
            (NodeChildrenRefMut::Inner(_), NodeEntry::Leaf(_)) => {
                panic!("cannot insert leaf entry in inner node")
            }
            (NodeChildrenRefMut::Leaf(_), NodeEntry::Inner(_)) => {
                panic!("cannot insert inner entry in leaf node")
            }
        }
    }

    fn insert(
        &mut self,
        entry: NodeEntry<N, D, Key, Value>,
    ) -> Option<NodeContainer<N, D, Key, Value>>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Bounded<N, D>,
    {
        let entry_bounds = entry.bounds();
        if entry.target_level() != self.level {
            let mut children = match self.children_mut() {
                NodeChildrenRefMut::Inner(children) => children,
                NodeChildrenRefMut::Leaf(_) => {
                    unreachable!()
                }
            };
            let mut insert_child =
                select::minimal_volume_increase(children.iter_mut(), &entry_bounds).unwrap();
            if let Some(new_child) = insert_child.insert(entry) {
                // The child node split, so the entries in new_child are no longer part of self
                // Recompute the bounds of self before trying to insert new_child into self
                self.node.bounds = Bounds::union_all(children.iter().map(|child| child.bounds()));
                self.self_insert(NodeEntry::Inner(new_child))
            } else {
                self.node.bounds = Bounds::union(&self.node.bounds, &entry_bounds);
                None
            }
        } else {
            self.self_insert(entry)
        }
    }

    fn insert_unique(
        &mut self,
        key: Key,
        value: Value,
    ) -> (Option<Value>, Option<NodeContainer<N, D, Key, Value>>)
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Eq + Bounded<N, D>,
    {
        let entry_bounds = key.bounds();
        match self.children_mut() {
            NodeChildrenRefMut::Inner(mut children) => {
                match get_only(
                    children
                        .iter_mut()
                        .filter(|child| child.bounds().contains(&key.bounds())),
                ) {
                    GetOnlyResult::None => {
                        // If there are no children containing the key bounds, then
                        // the key cannot exist in the node and it can be inserted.
                        (None, self.insert(NodeEntry::Leaf((key, value))))
                    }
                    GetOnlyResult::Only(mut child) => {
                        // If there is only one child containing the key bounds,
                        // then we can maintain the uniqueness invariant by
                        // performing a unique insert into that child, which may
                        // cause the child to split.
                        let (old_value, new_child) = child.insert_unique(key, value);
                        (
                            old_value,
                            if let Some(new_child) = new_child {
                                // The child node split, so the entries in new_child are no longer part of self
                                // Recompute the bounds of self before trying to insert new_child into self
                                self.node.bounds =
                                    Bounds::union_all(children.iter().map(|child| child.bounds()));
                                self.self_insert(NodeEntry::Inner(new_child))
                            } else {
                                self.node.bounds = Bounds::union(&self.node.bounds, &entry_bounds);
                                None
                            },
                        )
                    }
                    GetOnlyResult::Multiple => {
                        // Try to find the key among the children and overwrite it if it
                        // exists. If it doesn't exist, perform a regular insert.
                        if let Some(value_ref) = self.get_mut(&key) {
                            (Some(std::mem::replace(value_ref, value)), None)
                        } else {
                            (None, self.insert(NodeEntry::Leaf((key, value))))
                        }
                    }
                }
            }
            NodeChildrenRefMut::Leaf(mut children) => {
                // Try to find the key among the children and overwrite it if it
                // exists. If it doesn't exist, perform a regular insert.
                if let Some(entry) = children.iter_mut().find(|(k, _)| k == &key) {
                    (Some(std::mem::replace(&mut entry.1, value)), None)
                } else {
                    (None, self.insert(NodeEntry::Leaf((key, value))))
                }
            }
        }
    }

    fn into_children_mut(self) -> NodeChildrenRefMut<'a, N, D, Key, Value> {
        if let Some(level) = NonZeroUsize::new(self.level) {
            NodeChildrenRefMut::Inner(InnerNodeChildrenRefMut {
                ops: self.ops,
                level,
                children: unsafe { &mut self.node.children.inner },
            })
        } else {
            NodeChildrenRefMut::Leaf(unsafe {
                self.ops.children.wrap_ref_mut(&mut self.node.children.leaf)
            })
        }
    }

    fn children_mut<'b>(&'b mut self) -> NodeChildrenRefMut<'b, N, D, Key, Value> {
        if let Some(level) = NonZeroUsize::new(self.level) {
            NodeChildrenRefMut::Inner(InnerNodeChildrenRefMut {
                ops: self.ops,
                level,
                children: unsafe { &mut self.node.children.inner },
            })
        } else {
            NodeChildrenRefMut::Leaf(unsafe {
                self.ops.children.wrap_ref_mut(&mut self.node.children.leaf)
            })
        }
    }

    fn remove<Q>(
        &mut self,
        key: &Q,
        on_underfull: &mut impl FnMut(NodeChildrenContainer<N, D, Key, Value>),
    ) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq + Borrow<Q>,
        Q: Bounded<N, D> + Eq + ?Sized,
    {
        let min_children = self.ops.min_children;
        match self.children_mut() {
            NodeChildrenRefMut::Inner(mut children) => {
                let mut i = children.len();
                while i > 0 {
                    i -= 1;
                    let mut child = children.at_mut(i);
                    if child.bounds().intersects(&key.bounds()) {
                        let value = child.remove(key, on_underfull);
                        if let Some(value) = value {
                            if child.shallow_len() < min_children {
                                let child = children.swap_remove(i);
                                on_underfull(child.children());
                            }

                            self.node.bounds =
                                Bounds::union_all(children.iter().map(|child| child.bounds()));

                            return Some(value);
                        }
                    }
                }
                return None;
            }
            NodeChildrenRefMut::Leaf(mut children) => {
                let index = children.iter().position(|(k, _)| k.borrow() == key);
                if let Some(i) = index {
                    let value = children.swap_remove(i).1;
                    self.node.bounds =
                        Bounds::union_all(children.iter().map(|(key, _)| key.bounds()));

                    return Some(value);
                }
                return None;
            }
        }
    }

    pub(crate) fn into_get_mut<Q>(self, key: &Q) -> Option<&'a mut Value>
    where
        N: Ord,
        Key: Borrow<Q>,
        Q: Eq + Bounded<N, D> + ?Sized,
    {
        match self.into_children_mut() {
            NodeChildrenRefMut::Inner(children) => {
                for child in children {
                    if child.node.bounds.contains(&key.bounds()) {
                        if let Some(value) = child.into_get_mut(key) {
                            return Some(value);
                        }
                    }
                }
                None
            }
            NodeChildrenRefMut::Leaf(children) => children
                .into_iter()
                .find(|(k, _)| k.borrow() == key)
                .map(|(_, v)| v),
        }
    }

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut Value>
    where
        N: Ord,
        Key: Borrow<Q>,
        Q: Eq + Bounded<N, D> + ?Sized,
    {
        match self.children_mut() {
            NodeChildrenRefMut::Inner(children) => {
                for child in children {
                    if child.node.bounds.contains(&key.bounds()) {
                        if let Some(value) = child.into_get_mut(key) {
                            return Some(value);
                        }
                    }
                }
                None
            }
            NodeChildrenRefMut::Leaf(children) => children
                .into_iter()
                .find(|(k, _)| k.borrow() == key)
                .map(|(_, v)| v),
        }
    }

    pub(crate) unsafe fn drop(&mut self) {
        match self.children_mut() {
            NodeChildrenRefMut::Inner(children) => {
                for mut child in children {
                    child.drop();
                }
            }
            NodeChildrenRefMut::Leaf(_) => {}
        }
    }

    fn shallow_len(&self) -> usize {
        self.node.children.len(self.level)
    }
}

impl<'a, N, const D: usize, Key, Value> Bounded<N, D> for NodeRefMut<'a, N, D, Key, Value>
where
    N: Clone,
    Key: Bounded<N, D>,
{
    fn bounds(&self) -> Bounds<N, D> {
        self.node.bounds()
    }
}

impl<'a, N, const D: usize, Key, Value> From<&'a mut RootNodeRefMut<'_, N, D, Key, Value>>
    for NodeRefMut<'a, N, D, Key, Value>
{
    fn from(root: &'a mut RootNodeRefMut<'_, N, D, Key, Value>) -> Self {
        NodeRefMut {
            ops: root.ops,
            level: *root.height,
            node: root.node,
        }
    }
}

pub(crate) struct RootNodeRefMut<'a, N, const D: usize, Key, Value> {
    ops: NodeOps,
    height: &'a mut usize,
    node: &'a mut Node<N, D, Key, Value>,
}

impl<'a, N, const D: usize, Key, Value> RootNodeRefMut<'a, N, D, Key, Value> {
    fn branch(&mut self, sibling: NodeContainer<N, D, Key, Value>)
    where
        N: Ord + Clone + num_traits::Bounded,
    {
        let bounds = Bounds::union(&self.node.bounds, &sibling.node.bounds);
        let mut next_root_children = self.ops.children.new();
        unsafe {
            next_root_children.push(ptr::read(self.node));
            next_root_children.push(sibling.unwrap());
            ptr::write(
                self.node,
                Node::new(
                    bounds,
                    NodeChildren {
                        inner: ManuallyDrop::new(next_root_children.unwrap()),
                    },
                    *self.height,
                ),
            );
        }
        *self.height += 1;
    }

    fn insert_entry(&mut self, entry: NodeEntry<N, D, Key, Value>)
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Bounded<N, D>,
    {
        if let Some(sibling) = self.node_ref_mut().insert(entry) {
            self.branch(sibling);
        }
    }

    pub(crate) fn insert(&mut self, key: Key, value: Value)
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Bounded<N, D>,
    {
        self.insert_entry(NodeEntry::Leaf((key, value)));
    }

    pub(crate) fn insert_unique(&mut self, key: Key, value: Value) -> Option<Value>
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
        Key: Eq + Bounded<N, D>,
    {
        let (prev_value, sibling) = self.node_ref_mut().insert_unique(key, value);
        if let Some(sibling) = sibling {
            self.branch(sibling);
        }
        prev_value
    }

    fn node_ref_mut<'b>(&'b mut self) -> NodeRefMut<'b, N, D, Key, Value> {
        NodeRefMut {
            ops: self.ops,
            level: *self.height,
            node: self.node,
        }
    }

    fn try_unbranch(&mut self) {
        let new_root = match self.node_ref_mut().children_mut() {
            NodeChildrenRefMut::Inner(mut children) => {
                if children.len() == 1 {
                    Some(children.swap_remove(0))
                } else {
                    None
                }
            }
            NodeChildrenRefMut::Leaf(_) => {
                // Cannot unbranch a leaf node
                None
            }
        };
        if let Some(new_root) = new_root {
            unsafe {
                self.node_ref_mut().drop();
                *self.node = new_root.unwrap();
            }
            *self.height -= 1;
        }
    }

    pub(crate) fn remove<Q>(&'a mut self, key: &Q) -> Option<Value>
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq + Borrow<Q>,
        Q: Bounded<N, D> + Eq + ?Sized,
    {
        let mut reinsert_entries: Box<[Option<NodeChildren<N, D, Key, Value>>]> =
            std::iter::repeat_with(|| None).take(*self.height).collect();
        if let Some(value) = self.node_ref_mut().remove(key, &mut |children| {
            let level = children.level();
            reinsert_entries[level] = Some(unsafe { children.unwrap() });
        }) {
            self.try_unbranch();

            for (level, entries) in reinsert_entries.iter_mut().enumerate() {
                if let Some(entries) = entries.take() {
                    let entries = unsafe { self.ops.wrap_children(entries, level) };
                    match entries {
                        NodeChildrenContainer::Inner(children) => {
                            for entry in children {
                                self.insert_entry(NodeEntry::Inner(entry));
                            }
                        }
                        NodeChildrenContainer::Leaf(children) => {
                            for entry in children {
                                self.insert_entry(NodeEntry::Leaf(entry));
                            }
                        }
                    }
                }
            }

            Some(value)
        } else {
            None
        }
    }
}

pub(crate) struct NodeContainer<N, const D: usize, Key, Value> {
    ops: NodeOps,
    level: usize,
    node: Node<N, D, Key, Value>,
}

impl<'a, N, const D: usize, Key, Value> NodeContainer<N, D, Key, Value> {
    fn r#ref(&'a self) -> NodeRef<'a, N, D, Key, Value> {
        NodeRef {
            ops: self.ops,
            level: self.level,
            node: &self.node,
        }
    }

    fn ref_mut(&'a mut self) -> NodeRefMut<'a, N, D, Key, Value> {
        NodeRefMut {
            ops: self.ops,
            level: self.level,
            node: &mut self.node,
        }
    }

    fn children(self) -> NodeChildrenContainer<N, D, Key, Value> {
        let level = self.level;
        let ops = self.ops;
        let node = unsafe {
            let node = ptr::read(&self.node);
            mem::forget(self);
            node
        };
        if let Some(level) = NonZeroUsize::new(level) {
            NodeChildrenContainer::Inner(InnerNodeChildrenContainer {
                ops,
                level,
                children: ManuallyDrop::into_inner(unsafe { node.children.inner }),
            })
        } else {
            NodeChildrenContainer::Leaf(unsafe {
                ops.children
                    .wrap(ManuallyDrop::into_inner(node.children.leaf))
            })
        }
    }

    /// Unwraps the node from the container without dropping it.
    pub(crate) unsafe fn unwrap(self) -> Node<N, D, Key, Value> {
        let node = ptr::read(&self.node);
        mem::forget(self);
        node
    }
}

impl<N, const D: usize, Key, Value> Drop for NodeContainer<N, D, Key, Value> {
    fn drop(&mut self) {
        unsafe { self.ref_mut().drop() }
    }
}

impl<N, const D: usize, Key, Value> Clone for NodeContainer<N, D, Key, Value>
where
    N: Clone,
    Key: Clone,
    Value: Clone,
{
    fn clone(&self) -> Self {
        self.r#ref().clone()
    }
}

impl<N, const D: usize, Key, Value> Debug for NodeContainer<N, D, Key, Value>
where
    N: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("bounds", &self.node.bounds)
            .field("node", &self.r#ref().children())
            .finish()
    }
}

pub(crate) struct NodeRef<'a, N, const D: usize, Key, Value> {
    ops: NodeOps,
    level: usize,
    node: &'a Node<N, D, Key, Value>,
}

impl<'a, N, const D: usize, Key, Value> NodeRef<'a, N, D, Key, Value> {
    pub(crate) fn children(&self) -> NodeChildrenRef<'a, N, D, Key, Value> {
        if let Some(level) = NonZeroUsize::new(self.level) {
            NodeChildrenRef::Inner(InnerNodeChildrenRef {
                ops: self.ops,
                level,
                children: unsafe { &self.node.children.inner },
            })
        } else {
            NodeChildrenRef::Leaf(unsafe { self.ops.children.wrap_ref(&self.node.children.leaf) })
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self.children() {
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

    pub(crate) fn get<Q>(&self, key: &Q) -> Option<&'a Value>
    where
        N: Ord,
        Key: Borrow<Q>,
        Q: Eq + Bounded<N, D> + ?Sized,
    {
        match self.children() {
            NodeChildrenRef::Inner(children) => {
                for child in children {
                    if child.node.bounds.contains(&key.bounds()) {
                        if let Some(value) = child.get(key) {
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

    pub(crate) fn clone(&self) -> NodeContainer<N, D, Key, Value>
    where
        N: Clone,
        Key: Clone,
        Value: Clone,
    {
        let bounds = self.node.bounds.clone();
        let children = self.children().clone();
        unsafe {
            self.ops
                .wrap(Node::new(bounds, children.unwrap(), self.level), self.level)
        }
    }

    pub(crate) fn _debug_assert_bvh(&self) -> Bounds<N, D>
    where
        Key: Bounded<N, D>,
        N: Ord + num_traits::Bounded + Clone + Eq + Debug,
    {
        let bounds = match self.children() {
            NodeChildrenRef::Inner(children) => {
                Bounds::union_all(children.iter().map(|child| child._debug_assert_bvh()))
            }
            NodeChildrenRef::Leaf(children) => {
                Bounds::union_all(children.iter().map(|(key, _)| key.bounds()))
            }
        };

        assert_eq!(self.node.bounds, bounds);
        bounds
    }

    pub(crate) fn _debug_assert_eq(&self, other: &NodeRef<N, D, Key, Value>)
    where
        N: Debug + Eq,
        Key: Debug + Eq,
        Value: Debug + Eq,
    {
        assert_eq!(self.level, other.level);
        self.children()._debug_assert_eq(&other.children());
    }

    pub(crate) fn _debug_assert_min_children(&self, is_root: bool) {
        let children = self.children();
        if !is_root {
            assert!(children.len() >= self.ops.min_children);
        }
        if let NodeChildrenRef::Inner(children) = children {
            for child in children {
                child._debug_assert_min_children(false);
            }
        }
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
            .field("node", &self.children())
            .finish()
    }
}

impl<'a, N, const D: usize, Key, Value> Bounded<N, D> for NodeRef<'a, N, D, Key, Value>
where
    N: Clone,
    Key: Bounded<N, D>,
{
    fn bounds(&self) -> Bounds<N, D> {
        self.node.bounds()
    }
}

pub(crate) struct InnerNodeChildrenRef<'a, N, const D: usize, Key, Value> {
    ops: NodeOps,
    level: NonZeroUsize,
    children: &'a FCVec<Node<N, D, Key, Value>>,
}

impl<'a, N, const D: usize, Key, Value> From<InnerNodeChildrenRefMut<'a, N, D, Key, Value>>
    for InnerNodeChildrenRef<'a, N, D, Key, Value>
{
    fn from(children: InnerNodeChildrenRefMut<'a, N, D, Key, Value>) -> Self {
        InnerNodeChildrenRef {
            ops: children.ops,
            level: children.level,
            children: children.children,
        }
    }
}

impl<'a, N, const D: usize, Key, Value> Debug for InnerNodeChildrenRef<'a, N, D, Key, Value>
where
    N: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, N, const D: usize, Key, Value> InnerNodeChildrenRef<'a, N, D, Key, Value> {
    fn len(&self) -> usize {
        self.children.len()
    }

    fn iter(&self) -> InnerNodeChildrenIter<'a, N, D, Key, Value> {
        InnerNodeChildrenIter {
            ops: self.ops,
            level: self.level,
            children: self.children.iter(),
        }
    }

    fn clone(&self) -> InnerNodeChildrenContainer<N, D, Key, Value>
    where
        N: Clone,
        Key: Clone,
        Value: Clone,
    {
        let mut clone_children = self.ops.children.new();
        for child in self.iter() {
            clone_children.push(unsafe { child.clone().unwrap() });
        }

        InnerNodeChildrenContainer {
            ops: self.ops,
            level: self.level,
            children: unsafe { clone_children.unwrap() },
        }
    }

    fn _debug_assert_eq(&self, other: &InnerNodeChildrenRef<'a, N, D, Key, Value>)
    where
        N: Debug + Eq,
        Key: Debug + Eq,
        Value: Debug + Eq,
    {
        assert_eq!(self.level, other.level);
        assert_eq!(self.children.len(), other.children.len());
        for (child, other_child) in self.iter().zip(other.iter()) {
            child._debug_assert_eq(&other_child)
        }
    }
}

pub(crate) struct InnerNodeChildrenIter<'a, N, const D: usize, Key, Value> {
    ops: NodeOps,
    level: NonZeroUsize,
    children: slice::Iter<'a, Node<N, D, Key, Value>>,
}

impl<'a, N, const D: usize, Key, Value> IntoIterator
    for InnerNodeChildrenRef<'a, N, D, Key, Value>
{
    type Item = NodeRef<'a, N, D, Key, Value>;
    type IntoIter = InnerNodeChildrenIter<'a, N, D, Key, Value>;

    fn into_iter(self) -> Self::IntoIter {
        InnerNodeChildrenIter {
            ops: self.ops,
            level: self.level,
            children: self.children.iter(),
        }
    }
}

impl<'a, N, const D: usize, Key, Value> Iterator for InnerNodeChildrenIter<'a, N, D, Key, Value> {
    type Item = NodeRef<'a, N, D, Key, Value>;

    fn next(&mut self) -> Option<Self::Item> {
        self.children
            .next()
            .map(|node| unsafe { self.ops.wrap_ref(node, self.level.get() - 1) })
    }
}

pub(crate) enum NodeChildrenRef<'a, N, const D: usize, Key, Value> {
    Inner(InnerNodeChildrenRef<'a, N, D, Key, Value>),
    Leaf(FCVecRef<'a, (Key, Value)>),
}

impl<'a, N, const D: usize, Key, Value> Debug for NodeChildrenRef<'a, N, D, Key, Value>
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

impl<'a, N, const D: usize, Key, Value> NodeChildrenRef<'a, N, D, Key, Value> {
    fn len(&self) -> usize {
        match self {
            NodeChildrenRef::Inner(children) => children.len(),
            NodeChildrenRef::Leaf(children) => children.len(),
        }
    }

    fn clone(&self) -> NodeChildrenContainer<N, D, Key, Value>
    where
        N: Clone,
        Key: Clone,
        Value: Clone,
    {
        match self {
            NodeChildrenRef::Inner(children) => NodeChildrenContainer::Inner(children.clone()),
            NodeChildrenRef::Leaf(children) => NodeChildrenContainer::Leaf(children.clone()),
        }
    }

    fn _debug_assert_eq(&self, other: &NodeChildrenRef<N, D, Key, Value>)
    where
        N: Debug + Eq,
        Key: Debug + Eq,
        Value: Debug + Eq,
    {
        match (self, other) {
            (NodeChildrenRef::Inner(children), NodeChildrenRef::Inner(other_children)) => {
                children._debug_assert_eq(other_children)
            }
            (NodeChildrenRef::Leaf(children), NodeChildrenRef::Leaf(other_children)) => {
                assert_eq!(**children, **other_children)
            }
            _ => panic!("Cannot compare inner and leaf node children"),
        }
    }
}

struct InnerNodeChildrenRefMut<'a, N, const D: usize, Key, Value> {
    ops: NodeOps,
    level: NonZeroUsize,
    children: &'a mut FCVec<Node<N, D, Key, Value>>,
}

impl<'a, N, const D: usize, Key, Value> InnerNodeChildrenRefMut<'a, N, D, Key, Value> {
    unsafe fn drop(&mut self) {
        for mut child in self.iter_mut() {
            child.drop();
        }
    }

    fn len(&self) -> usize {
        self.children.len()
    }

    fn at_mut<'b>(&'b mut self, index: usize) -> NodeRefMut<'b, N, D, Key, Value> {
        unsafe {
            self.ops
                .wrap_ref_mut(&mut self.children[index], self.level.get() - 1)
        }
    }

    fn swap_remove(&mut self, index: usize) -> NodeContainer<N, D, Key, Value> {
        unsafe {
            self.ops
                .wrap(self.children.swap_remove(index), self.level.get() - 1)
        }
    }

    fn iter<'b>(&'b self) -> InnerNodeChildrenIter<'b, N, D, Key, Value> {
        InnerNodeChildrenIter {
            ops: self.ops,
            level: self.level,
            children: self.children.iter(),
        }
    }

    fn iter_mut<'b>(&'b mut self) -> InnerNodeChildrenIterMut<'b, N, D, Key, Value> {
        InnerNodeChildrenIterMut {
            ops: self.ops,
            level: self.level,
            children: self.children.iter_mut(),
        }
    }

    fn try_push(
        &mut self,
        node: NodeContainer<N, D, Key, Value>,
    ) -> Option<NodeContainer<N, D, Key, Value>> {
        if node.level != self.level.get() - 1 {
            panic!("Cannot push a node with the wrong level");
        }
        let mut children = unsafe { self.ops.children.wrap_ref_mut(&mut self.children) };
        children
            .try_push(unsafe { node.unwrap() })
            .map(|node| unsafe { self.ops.wrap(node, self.level.get() - 1) })
    }

    fn split(
        &mut self,
        overflow_node: NodeContainer<N, D, Key, Value>,
    ) -> (Bounds<N, D>, NodeContainer<N, D, Key, Value>)
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64>,
    {
        let children = unsafe { self.ops.children.wrap_ref_mut(&mut self.children) };
        let (new_bounds, sibling_bounds, sibling_children) =
            unsafe { split::quadratic(self.ops.min_children, children, overflow_node.unwrap()) };
        (new_bounds, unsafe {
            self.ops.wrap(
                Node::new(
                    sibling_bounds,
                    NodeChildren {
                        inner: ManuallyDrop::new(sibling_children.unwrap()),
                    },
                    self.level.get(),
                ),
                self.level.get(),
            )
        })
    }
}

struct InnerNodeChildrenIterMut<'a, N, const D: usize, Key, Value> {
    ops: NodeOps,
    level: NonZeroUsize,
    children: slice::IterMut<'a, Node<N, D, Key, Value>>,
}

impl<'a, N, const D: usize, Key, Value> IntoIterator
    for InnerNodeChildrenRefMut<'a, N, D, Key, Value>
{
    type Item = NodeRefMut<'a, N, D, Key, Value>;
    type IntoIter = InnerNodeChildrenIterMut<'a, N, D, Key, Value>;

    fn into_iter(self) -> Self::IntoIter {
        InnerNodeChildrenIterMut {
            ops: self.ops,
            level: self.level,
            children: self.children.iter_mut(),
        }
    }
}

impl<'a, N, const D: usize, Key, Value> Iterator
    for InnerNodeChildrenIterMut<'a, N, D, Key, Value>
{
    type Item = NodeRefMut<'a, N, D, Key, Value>;

    fn next(&mut self) -> Option<Self::Item> {
        self.children
            .next()
            .map(|node| unsafe { self.ops.wrap_ref_mut(node, self.level.get() - 1) })
    }
}

enum NodeChildrenRefMut<'a, N, const D: usize, Key, Value> {
    Inner(InnerNodeChildrenRefMut<'a, N, D, Key, Value>),
    Leaf(FCVecRefMut<'a, (Key, Value)>),
}

struct InnerNodeChildrenContainer<N, const D: usize, Key, Value> {
    ops: NodeOps,
    level: NonZeroUsize,
    children: FCVec<Node<N, D, Key, Value>>,
}

impl<N, const D: usize, Key, Value> IntoIterator for InnerNodeChildrenContainer<N, D, Key, Value> {
    type Item = NodeContainer<N, D, Key, Value>;
    type IntoIter = InnerNodeChildrenIntoIter<N, D, Key, Value>;

    fn into_iter(self) -> Self::IntoIter {
        let ops = self.ops;
        let level = self.level;
        let children = unsafe { self.unwrap() };
        InnerNodeChildrenIntoIter {
            ops,
            level,
            children: unsafe { ops.children.wrap(children) }.into_iter(),
        }
    }
}

impl<'a, N, const D: usize, Key, Value> Drop for InnerNodeChildrenContainer<N, D, Key, Value> {
    fn drop(&mut self) {
        unsafe { self.ref_mut().drop() }
    }
}

impl<'a, N, const D: usize, Key, Value> InnerNodeChildrenContainer<N, D, Key, Value> {
    fn ref_mut(&'a mut self) -> InnerNodeChildrenRefMut<'a, N, D, Key, Value> {
        InnerNodeChildrenRefMut {
            ops: self.ops,
            level: self.level,
            children: &mut self.children,
        }
    }

    unsafe fn unwrap(self) -> FCVec<Node<N, D, Key, Value>> {
        let children = ptr::read(&self.children);
        mem::forget(self);
        children
    }
}

struct InnerNodeChildrenIntoIter<N, const D: usize, Key, Value> {
    ops: NodeOps,
    level: NonZeroUsize,
    children: fc_vec::IntoIter<Node<N, D, Key, Value>>,
}

impl<N, const D: usize, Key, Value> Iterator for InnerNodeChildrenIntoIter<N, D, Key, Value> {
    type Item = NodeContainer<N, D, Key, Value>;

    fn next(&mut self) -> Option<Self::Item> {
        self.children
            .next()
            .map(|node| unsafe { self.ops.wrap(node, self.level.get() - 1) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.children.size_hint()
    }
}

impl<N, const D: usize, Key, Value> Drop for InnerNodeChildrenIntoIter<N, D, Key, Value> {
    fn drop(&mut self) {
        for _ in &mut *self {}
    }
}

enum NodeChildrenContainer<N, const D: usize, Key, Value> {
    Inner(InnerNodeChildrenContainer<N, D, Key, Value>),
    Leaf(FCVecContainer<(Key, Value)>),
}

impl<N, const D: usize, Key, Value> NodeChildrenContainer<N, D, Key, Value> {
    fn level(&self) -> usize {
        match self {
            NodeChildrenContainer::Inner(children) => children.level.get(),
            NodeChildrenContainer::Leaf(_) => 0,
        }
    }

    unsafe fn unwrap(self) -> NodeChildren<N, D, Key, Value> {
        match self {
            NodeChildrenContainer::Inner(children) => NodeChildren {
                inner: ManuallyDrop::new(children.unwrap()),
            },
            NodeChildrenContainer::Leaf(children) => NodeChildren {
                leaf: ManuallyDrop::new(children.unwrap()),
            },
        }
    }
}
