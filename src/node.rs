use std::{
    borrow::Borrow,
    fmt::Debug,
    mem::{self, ManuallyDrop},
    num::NonZeroUsize,
    ptr, slice,
};

use crate::{
    bounds::{Bounded, Bounds},
    contains::Contains,
    fc_vec::{self, FCVec, FCVecContainer, FCVecRefMut},
    intersects::Intersects,
    select::Selector,
    split::Splitter,
    util::{get_only, GetOnlyResult},
};

pub(crate) enum NodeEntry<B, Key, Value> {
    Inner(NodeContainer<B, Key, Value>),
    Leaf((Key, Value)),
}

impl<B, Key, Value> NodeEntry<B, Key, Value> {
    fn target_level(&self) -> usize {
        match self {
            NodeEntry::Inner(node) => node.level + 1,
            NodeEntry::Leaf(_) => 0,
        }
    }
}

impl<B, Key, Value> Bounded<B> for NodeEntry<B, Key, Value>
where
    B: Clone,
    Key: Bounded<B>,
{
    fn bounds(&self) -> B {
        match self {
            NodeEntry::Inner(node_container) => node_container.node.bounds(),
            NodeEntry::Leaf((key, _)) => key.bounds(),
        }
    }
}

union NodeChildren<B, Key, Value> {
    inner: ManuallyDrop<FCVec<Node<B, Key, Value>>>,
    leaf: ManuallyDrop<FCVec<(Key, Value)>>,
}

/// A Node is either an inner node or a leaf node. The children of an inner node
/// is a vector of Nodes, while the children of a leaf node is a vector of
/// key-value pairs. The bounds of a Node is the union of the bounds of its
/// children.
///
/// A Node is not self-describing in the sense that it has no explicit level or
/// capacity. These must be provided on creation and provided consistently when
/// manipulating the node. This is done to avoid storing data that is constant
/// for sets of nodes, or data that is easily computed from other data.
///
/// Being non-self-describing makes Node a fundamentally unsafe type, but it may
/// be used safely by wrapping it in a [`NodeContainer`], a [`NodeRef`] or a
/// [`NodeRefMut`] using the [`Alloc`] that created it. Wrapping the node
/// requires knowledge of the implicit level of the node.
///
/// Node does not implement [`Drop`], but to avoid leaks it must be dropped
/// either by wrapping it in a [`NodeContainer`] or by wrapping it in a
/// [`NodeRefMut`] and calling [`NodeRefMut::drop`].
///
/// # Implicit parameters
///
/// A Node has an implicit level which is provided as the `_level` parameter to
/// the constructor. It is implicit because it is not stored in the node and
/// must be tracked separately. The level describes the node's position in the
/// tree. A node with level zero is a leaf node, while a node with a positive
/// level `l` is an inner node containing nodes with level `l - 1`.
///
/// A Node also has a capacity determined by the [`Alloc`] used to create it.
/// Like the level, the capacity is not stored in the Node, and is instead
/// maintained by consistently using the same [`Alloc`] to manipulate the Node.
pub struct Node<B, Key, Value> {
    pub(crate) bounds: B,
    children: NodeChildren<B, Key, Value>,
}

impl<B, Key, Value> Node<B, Key, Value> {
    /// Creates a new node. The bounds must be the union of the bounds of the
    /// children. It is a logic error to create a node with bounds that are not
    /// the union of the bounds of the children.
    ///
    /// # Safety
    ///
    /// `_level` describes the implicit level of the node. Though it is not
    /// used to produce the node, it is provided as a parameter for debugging
    /// purposes. It is the caller's responsibility to ensure that same level is
    /// provided for operations on the node that depend on the level.
    ///
    /// `children` must be set appropriately based on `_level`. If `_level` = 0,
    /// `children` must be leaf children. If `_level` > 0, `children` must be
    /// inner children containing nodes whose level is `_level - 1`. It is
    /// undefined behavior to violate these conditions.
    unsafe fn new(bounds: B, children: NodeChildren<B, Key, Value>, _level: usize) -> Self {
        Node { bounds, children }
    }

    /// Returns the children of the node as an inner node, which is a vector of
    /// nodes. For a node with implicit level `l`, the level of the children is
    /// `l - 1`.
    ///
    /// # Safety
    ///
    /// The node must be an inner node (a node with level > 0). It is undefined
    /// behavior to call this method on a leaf node.
    pub(crate) unsafe fn inner_children(&self) -> &FCVec<Node<B, Key, Value>> {
        // SAFETY: The node is an inner node, so the children are initialized as
        // inner children.
        &self.children.inner
    }

    /// Returns the children of the node as an inner node, which is a vector of
    /// nodes. For a node with implicit level `l`, the level of the children is
    /// `l - 1`.
    ///
    /// # Safety
    ///
    /// The node must be an inner node (a node with level > 0). It is undefined
    /// behavior to call this method on a leaf node.
    pub(crate) unsafe fn inner_children_mut(&mut self) -> &mut FCVec<Node<B, Key, Value>> {
        // SAFETY: The node is an inner node, so the children are initialized as
        // inner children.
        &mut self.children.inner
    }

    /// Returns the children of the node as a leaf node, which is a vector of
    /// key-value pairs.
    ///
    /// # Safety
    ///
    /// The node must be a leaf node (a node with level = 0). It is undefined
    /// behavior to call this method on an inner node.
    pub(crate) unsafe fn leaf_children(&self) -> &FCVec<(Key, Value)> {
        // SAFETY: The node is a leaf node, so the children are initialized as
        // leaf children.
        &self.children.leaf
    }

    /// Returns the children of the node as a leaf node, which is a vector of
    /// key-value pairs.
    ///
    /// # Safety
    ///
    /// The node must be a leaf node (a node with level = 0). It is undefined
    /// behavior to call this method on an inner node.
    pub(crate) unsafe fn leaf_children_mut(&mut self) -> &mut FCVec<(Key, Value)> {
        // SAFETY: The node is a leaf node, so the children are initialized as
        // leaf children.
        &mut self.children.leaf
    }
}

impl<B, Key, Value> Bounded<B> for (Key, Value)
where
    Key: Bounded<B>,
{
    fn bounds(&self) -> B {
        self.0.bounds()
    }
}

impl<B, Key, Value> Bounded<B> for Node<B, Key, Value>
where
    B: Clone,
{
    fn bounds(&self) -> B {
        self.bounds.clone()
    }
}

#[derive(Copy, Clone)]
pub(crate) struct Alloc {
    children: fc_vec::Alloc,
    min_children: usize,
}

impl Alloc {
    pub(crate) fn new_alloc(min_children: usize, max_children: usize) -> Self {
        Alloc {
            children: fc_vec::Alloc::new_alloc(max_children),
            min_children,
        }
    }

    pub(crate) fn new_leaf<B, Key, Value>(&self) -> NodeContainer<B, Key, Value>
    where
        B: Bounds,
    {
        // SAFETY: The node is a leaf node, so its level is 0 and its children
        // are initialized as leaf children.
        unsafe {
            self.wrap(
                Node::new(
                    Bounds::empty(),
                    NodeChildren {
                        leaf: ManuallyDrop::new(self.children.new().leak()),
                    },
                    0,
                ),
                0,
            )
        }
    }

    /// Wraps a node with a provided level in a safe container that takes
    /// ownership of the node. The provided node must have been allocated by the
    /// same Alloc and must not have been dropped.
    ///
    /// # Safety
    ///
    /// The provided `level` must match the implicit level of the node. It is
    /// undefined behavior to call this method with a level that does not match
    /// the implicit level of the node.
    ///
    /// The method should not be called with a node that has been allocated by a
    /// different Alloc. It is undefined behavior to do so if the Allocs have
    /// different `max_children` values, and it is a logic error to do so if the
    /// Allocs have different `min_children` values.
    unsafe fn wrap<B, Key, Value>(
        &self,
        node: Node<B, Key, Value>,
        level: usize,
    ) -> NodeContainer<B, Key, Value> {
        NodeContainer {
            alloc: *self,
            level,
            node,
        }
    }

    /// Mutably borrows a node with a provided level in a safe reference. The
    /// provided `node` must have been allocated by the same Alloc and must not
    /// have been dropped.
    ///
    /// # Safety
    ///
    /// The provided `level` must match the implicit level of the node. It is
    /// undefined behavior to use the returned reference if this condition is
    /// violated.
    ///
    /// The return value must not be used if this is called with a node that has
    /// been allocated by a different Alloc. It is undefined behavior to do so
    /// if the Allocs have different `max_children` values, and it is a logic
    /// error to do so if the Allocs have different `min_children` values.
    pub(crate) unsafe fn wrap_ref_mut<'a, B, Key, Value>(
        &self,
        node: &'a mut Node<B, Key, Value>,
        level: usize,
    ) -> NodeRefMut<'a, B, Key, Value> {
        NodeRefMut {
            alloc: *self,
            level,
            node,
        }
    }

    /// Mutably borrows the root node and height of a tree in a safe reference.
    /// This is a special case of `wrap_ref_mut` that can be used for operations
    /// that can change the height of the tree. The provided `node` must have
    /// been allocated by the same Alloc and must not have been dropped.
    ///
    /// The provided height is equal to the level of the root node. If the
    /// returned reference is used for operations that can change the height of
    /// the tree, the provided `height` is updated to match the new height, and
    /// the root node is modified such that its implicit level matches the new
    /// height.
    ///
    /// # Safety
    ///
    /// The provided `height` must match the implicit height of the tree rooted
    /// at `node`. It is undefined behavior to use the returned reference if
    /// this condition is violated.
    ///
    /// The return value must not be used if this is called with a node that has
    /// been allocated by a different Alloc. It is undefined behavior to do so
    /// if the Allocs have different `max_children` values, and it is a logic
    /// error to do so if the Allocs have different `min_children` values.
    pub(crate) unsafe fn wrap_root_ref_mut<'a, B, Key, Value>(
        &self,
        node: &'a mut Node<B, Key, Value>,
        height: &'a mut usize,
    ) -> RootNodeRefMut<'a, B, Key, Value> {
        RootNodeRefMut {
            alloc: *self,
            height,
            node,
        }
    }

    /// Wraps the children of a node at a specified level in a safe container
    /// that takes ownership of the children, which will drop the children
    /// when dropped. The provided `children` must have been allocated by the
    /// same Alloc and must not have been dropped.
    ///
    /// # Safety
    ///
    /// The provided `level` must match the implicit level of the node that
    /// contained the children. It is undefined behavior to call this method
    /// with a level that does not match the implicit level of the node that
    /// contained the children.
    ///
    /// The method should not be called with children that were contained by a
    /// node that has been allocated by a different Alloc. It is undefined
    /// behavior to do so if the Alloc have different `max_children` values,
    /// and it is a logic error to do so if the Allocs have different
    /// `min_children` values.
    unsafe fn wrap_children<B, Key, Value>(
        &self,
        children: NodeChildren<B, Key, Value>,
        level: usize,
    ) -> NodeChildrenContainer<B, Key, Value> {
        if let Some(level) = NonZeroUsize::new(level) {
            NodeChildrenContainer::Inner(InnerNodeChildrenContainer {
                alloc: *self,
                level,
                children: ManuallyDrop::into_inner(children.inner),
            })
        } else {
            NodeChildrenContainer::Leaf(self.children.wrap(ManuallyDrop::into_inner(children.leaf)))
        }
    }

    pub(crate) unsafe fn clone_node<B, Key, Value>(
        &self,
        node: NodeRef<B, Key, Value>,
    ) -> NodeContainer<B, Key, Value>
    where
        B: Clone,
        Key: Clone,
        Value: Clone,
    {
        let bounds = node.node.bounds.clone();
        // let children = self.wrap_children(node.node.children, node.level);
        let children = self.clone_children(node.children());
        // SAFETY: `children` contains the same children as `self`, so the
        // level of the children is the same as the level of `self`. The clones
        // are allocated by the same fc_vec::Alloc as `self`.
        unsafe { self.wrap(Node::new(bounds, children.leak(), node.level), node.level) }
    }

    unsafe fn clone_children<B, Key, Value>(
        &self,
        children: NodeChildrenRef<B, Key, Value>,
    ) -> NodeChildrenContainer<B, Key, Value>
    where
        B: Clone,
        Key: Clone,
        Value: Clone,
    {
        match children {
            NodeChildrenRef::Inner(children) => {
                let level = children.level;
                let mut clone_children = self.children.new::<Node<B, Key, Value>>();
                for child in children {
                    clone_children.push(self.clone_node(child).leak());
                }
                NodeChildrenContainer::Inner(InnerNodeChildrenContainer {
                    alloc: *self,
                    level,
                    children: clone_children.leak(),
                })
            }
            NodeChildrenRef::Leaf(children) => {
                let children = self.children.clone(children);
                NodeChildrenContainer::Leaf(children)
            }
        }
    }
}

pub(crate) struct NodeRefMut<'a, B, Key, Value> {
    alloc: Alloc,
    level: usize,
    node: &'a mut Node<B, Key, Value>,
}

impl<'a, B, Key, Value> NodeRefMut<'a, B, Key, Value>
where
    B: Bounds + Clone,
    Key: Bounded<B>,
{
    fn self_insert(
        &mut self,
        splitter: &mut impl Splitter<B>,
        entry: NodeEntry<B, Key, Value>,
    ) -> Option<NodeContainer<B, Key, Value>> {
        let entry_bounds = entry.bounds();
        let min_children = self.alloc.min_children;
        match (self.children_mut(), entry) {
            (NodeChildrenRefMut::Inner(mut children), NodeEntry::Inner(entry)) => {
                if children.level.get() - 1 != entry.level {
                    panic!(
                        "cannot insert entry with level {} in node with level {}",
                        entry.level, children.level
                    );
                }
                if let Some(overflow_node) = children.try_push(entry) {
                    let (new_bounds, sibling) = children.split(splitter, overflow_node);
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
                        splitter.split(min_children, children, overflow_entry);
                    self.node.bounds = new_bounds;
                    // SAFETY: `sibling_children` is allocated by the same
                    // fc_vec::Alloc as `children`. The level of the sibling node is
                    // the same as the level of the current node. self.node is a
                    // leaf node, so the children of the sibling node are
                    // initialized as leaf children.
                    Some(unsafe {
                        self.alloc.wrap(
                            Node::new(
                                sibling_bounds,
                                NodeChildren {
                                    leaf: ManuallyDrop::new(sibling_children.leak()),
                                },
                                self.level,
                            ),
                            self.level,
                        )
                    })
                } else {
                    self.node.bounds = B::union(&self.node.bounds, &entry_bounds);
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
        selector: &mut impl Selector<B>,
        splitter: &mut impl Splitter<B>,
        entry: NodeEntry<B, Key, Value>,
    ) -> Option<NodeContainer<B, Key, Value>> {
        let entry_bounds = entry.bounds();
        if entry.target_level() != self.level {
            let mut children = match self.children_mut() {
                NodeChildrenRefMut::Inner(children) => children,
                NodeChildrenRefMut::Leaf(_) => {
                    unreachable!()
                }
            };
            let mut insert_child = selector.select(children.iter_mut(), &entry_bounds).unwrap();
            if let Some(new_child) = insert_child.insert(selector, splitter, entry) {
                // The child node split, so the entries in new_child are no longer part of self
                // Recompute the bounds of self before trying to insert new_child into self
                self.node.bounds = B::union_all(children.iter().map(|child| child.bounds()));
                self.self_insert(splitter, NodeEntry::Inner(new_child))
            } else {
                self.node.bounds = B::union(&self.node.bounds, &entry_bounds);
                None
            }
        } else {
            self.self_insert(splitter, entry)
        }
    }

    fn insert_unique(
        &mut self,
        selector: &mut impl Selector<B>,
        splitter: &mut impl Splitter<B>,
        key: Key,
        value: Value,
    ) -> (Option<Value>, Option<NodeContainer<B, Key, Value>>)
    where
        Key: Eq,
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
                        (
                            None,
                            self.insert(selector, splitter, NodeEntry::Leaf((key, value))),
                        )
                    }
                    GetOnlyResult::Only(mut child) => {
                        // If there is only one child containing the key bounds,
                        // then we can maintain the uniqueness invariant by
                        // performing a unique insert into that child, which may
                        // cause the child to split.
                        let (old_value, new_child) =
                            child.insert_unique(selector, splitter, key, value);
                        (
                            old_value,
                            if let Some(new_child) = new_child {
                                // The child node split, so the entries in new_child are no longer part of self
                                // Recompute the bounds of self before trying to insert new_child into self
                                self.node.bounds =
                                    B::union_all(children.iter().map(|child| child.bounds()));
                                self.self_insert(splitter, NodeEntry::Inner(new_child))
                            } else {
                                self.node.bounds = B::union(&self.node.bounds, &entry_bounds);
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
                            (
                                None,
                                self.insert(selector, splitter, NodeEntry::Leaf((key, value))),
                            )
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
                    (
                        None,
                        self.insert(selector, splitter, NodeEntry::Leaf((key, value))),
                    )
                }
            }
        }
    }

    fn remove<Q>(
        &mut self,
        key: &Q,
        on_underfull: &mut impl FnMut(NodeChildrenContainer<B, Key, Value>),
    ) -> Option<Value>
    where
        B: Intersects<B>,
        Key: Eq + Borrow<Q>,
        Q: Bounded<B> + Eq + ?Sized,
    {
        let min_children = self.alloc.min_children;
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
                                B::union_all(children.iter().map(|child| child.bounds()));

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
                    self.node.bounds = B::union_all(children.iter().map(|(key, _)| key.bounds()));

                    return Some(value);
                }
                return None;
            }
        }
    }
}

impl<'a, B, Key, Value> NodeRefMut<'a, B, Key, Value> {
    fn into_children_mut(self) -> NodeChildrenRefMut<'a, B, Key, Value> {
        if let Some(level) = NonZeroUsize::new(self.level) {
            NodeChildrenRefMut::Inner(InnerNodeChildrenRefMut {
                alloc: self.alloc,
                level,
                // SAFETY: The node is an inner node, so the children are
                // initialized as inner children.
                children: unsafe { &mut self.node.children.inner },
            })
        } else {
            // SAFETY: The node is a leaf node, so the children are initialized
            // as leaf children. `self.alloc.children` was used to create the
            // node's children.
            NodeChildrenRefMut::Leaf(unsafe {
                self.alloc
                    .children
                    .wrap_ref_mut(&mut self.node.children.leaf)
            })
        }
    }

    fn children_mut<'b>(&'b mut self) -> NodeChildrenRefMut<'b, B, Key, Value> {
        if let Some(level) = NonZeroUsize::new(self.level) {
            NodeChildrenRefMut::Inner(InnerNodeChildrenRefMut {
                alloc: self.alloc,
                level,
                // SAFETY: The node is an inner node, so the children are
                // initialized as inner children.
                children: unsafe { &mut self.node.children.inner },
            })
        } else {
            // SAFETY: The node is a leaf node, so the children are initialized
            // as leaf children. `self.alloc.children` was used to create the
            // node's children.
            NodeChildrenRefMut::Leaf(unsafe {
                self.alloc
                    .children
                    .wrap_ref_mut(&mut self.node.children.leaf)
            })
        }
    }

    /// Drops the node and all of its children. After calling this method, the
    /// node and references to it are invalid and must not be used.
    ///
    /// # Safety
    ///
    /// It is undefined behavior to drop a node that has already been dropped.
    ///
    /// It is undefined behavior to use a node after dropping it.
    pub(crate) unsafe fn drop(mut self) {
        match self.children_mut() {
            NodeChildrenRefMut::Inner(children) => {
                children.drop();
            }
            NodeChildrenRefMut::Leaf(children) => {
                children.drop();
            }
        }
    }

    fn shallow_len(&self) -> usize {
        if self.level > 0 {
            // SAFETY: The node is an inner node, so the children are
            // initialized as inner children.
            unsafe { (*self.node.children.inner).len() }
        } else {
            // SAFETY: The node is a leaf node, so the children are
            // initialized as leaf children.
            unsafe { (*self.node.children.leaf).len() }
        }
    }

    pub(crate) fn into_get_mut<Q>(self, key: &Q) -> Option<&'a mut Value>
    where
        B: Contains<B>,
        Key: Borrow<Q>,
        Q: Eq + Bounded<B> + ?Sized,
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
        B: Contains<B>,
        Key: Borrow<Q>,
        Q: Eq + Bounded<B> + ?Sized,
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
}

impl<'a, B, Key, Value> Bounded<B> for NodeRefMut<'a, B, Key, Value>
where
    B: Clone,
{
    fn bounds(&self) -> B {
        self.node.bounds()
    }
}

impl<'a, B, Key, Value> From<&'a mut RootNodeRefMut<'_, B, Key, Value>>
    for NodeRefMut<'a, B, Key, Value>
{
    fn from(root: &'a mut RootNodeRefMut<'_, B, Key, Value>) -> Self {
        NodeRefMut {
            alloc: root.alloc,
            level: *root.height,
            node: root.node,
        }
    }
}

pub(crate) struct RootNodeRefMut<'a, B, Key, Value> {
    alloc: Alloc,
    height: &'a mut usize,
    node: &'a mut Node<B, Key, Value>,
}

impl<'a, B, Key, Value> RootNodeRefMut<'a, B, Key, Value>
where
    B: Bounds + Clone,
    Key: Bounded<B>,
{
    fn insert_entry(
        &mut self,
        selector: &mut impl Selector<B>,
        splitter: &mut impl Splitter<B>,
        entry: NodeEntry<B, Key, Value>,
    ) {
        if let Some(sibling) = self.node_ref_mut().insert(selector, splitter, entry) {
            self.branch(sibling);
        }
    }

    pub(crate) fn insert(
        &mut self,
        selector: &mut impl Selector<B>,
        splitter: &mut impl Splitter<B>,
        key: Key,
        value: Value,
    ) {
        self.insert_entry(selector, splitter, NodeEntry::Leaf((key, value)));
    }

    pub(crate) fn insert_unique(
        &mut self,
        selector: &mut impl Selector<B>,
        splitter: &mut impl Splitter<B>,
        key: Key,
        value: Value,
    ) -> Option<Value>
    where
        Key: Eq,
    {
        let (prev_value, sibling) = self
            .node_ref_mut()
            .insert_unique(selector, splitter, key, value);
        if let Some(sibling) = sibling {
            self.branch(sibling);
        }
        prev_value
    }

    pub(crate) fn remove<Q>(
        &'a mut self,
        selector: &mut impl Selector<B>,
        splitter: &mut impl Splitter<B>,
        key: &Q,
    ) -> Option<Value>
    where
        B: Intersects<B>,
        Key: Eq + Borrow<Q>,
        Q: Bounded<B> + Eq + ?Sized,
    {
        let mut reinsert_entries: Box<[Option<NodeChildren<B, Key, Value>>]> =
            std::iter::repeat_with(|| None).take(*self.height).collect();
        if let Some(value) = self.node_ref_mut().remove(key, &mut |children| {
            let level = children.level();
            reinsert_entries[level] = Some(children.leak());
        }) {
            self.try_unbranch();

            for (level, entries) in reinsert_entries.iter_mut().enumerate() {
                if let Some(entries) = entries.take() {
                    // SAFETY: The children in `reinsert_entries` are placed in
                    // the array such that the index of each child is equal to
                    // the level of the node that contained the children. All
                    // children in the array are allocated by the same Alloc as
                    // the root.
                    let entries = unsafe { self.alloc.wrap_children(entries, level) };
                    match entries {
                        NodeChildrenContainer::Inner(children) => {
                            for entry in children {
                                self.insert_entry(selector, splitter, NodeEntry::Inner(entry));
                            }
                        }
                        NodeChildrenContainer::Leaf(children) => {
                            for entry in children {
                                self.insert_entry(selector, splitter, NodeEntry::Leaf(entry));
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

impl<'a, B, Key, Value> RootNodeRefMut<'a, B, Key, Value> {
    fn branch(&mut self, sibling: NodeContainer<B, Key, Value>)
    where
        B: Bounds,
    {
        if sibling.level != *self.height {
            panic!(
                "cannot branch with sibling of level {} when root node has level {}",
                sibling.level, *self.height
            )
        }

        let bounds = B::union(&self.node.bounds, &sibling.node.bounds);
        let mut next_root_children = self.alloc.children.new();
        // The below operation redefines the root node to be an inner node that
        // contains the former root node and the provided sibling. This requires
        // moving the root into itself.

        // SAFETY: self.node is temporarily made an invalid copy with ptr::read
        // but the invalid copy is subsequently overwritten with ptr::write. The
        // children of the new root node are both nodes at the same level as the
        // former root node, so the level of the new root node is the same as
        // the level of the former root node plus one.
        unsafe {
            next_root_children.push(ptr::read(self.node));
            next_root_children.push(sibling.leak());
            ptr::write(
                self.node,
                Node::new(
                    bounds,
                    NodeChildren {
                        inner: ManuallyDrop::new(next_root_children.leak()),
                    },
                    *self.height + 1,
                ),
            );
        }
        *self.height += 1;
    }

    fn node_ref_mut<'b>(&'b mut self) -> NodeRefMut<'b, B, Key, Value> {
        NodeRefMut {
            alloc: self.alloc,
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
            // SAFETY: The former root is replaced immediately after the former
            // root is dropped, so the former root is not used after being
            // dropped.
            unsafe {
                self.node_ref_mut().drop();
            }
            *self.node = new_root.leak();
            *self.height -= 1;
        }
    }
}

pub(crate) struct NodeContainer<B, Key, Value> {
    alloc: Alloc,
    level: usize,
    node: Node<B, Key, Value>,
}

impl<'a, B, Key, Value> NodeContainer<B, Key, Value> {
    fn borrow(&self) -> NodeRef<B, Key, Value> {
        NodeRef {
            level: self.level,
            node: &self.node,
        }
    }

    fn borrow_mut(&mut self) -> NodeRefMut<B, Key, Value> {
        NodeRefMut {
            alloc: self.alloc,
            level: self.level,
            node: &mut self.node,
        }
    }

    fn children(self) -> NodeChildrenContainer<B, Key, Value> {
        let level = self.level;
        let alloc = self.alloc;
        let node = {
            // SAFETY: self.node is not read after being moved out of the
            // container - self is dropped immediately afterwards.
            let node = unsafe { ptr::read(&self.node) };
            mem::forget(self);
            node
        };
        if let Some(level) = NonZeroUsize::new(level) {
            NodeChildrenContainer::Inner(InnerNodeChildrenContainer {
                alloc,
                level,
                // SAFETY: The node is an inner node, so the children are
                // initialized as inner children.
                children: ManuallyDrop::into_inner(unsafe { node.children.inner }),
            })
        } else {
            // SAFETY: The node is a leaf node, so the children are initialized
            // as leaf children. `self.alloc.children` was used to create the
            // node's children.
            NodeChildrenContainer::Leaf(unsafe {
                alloc
                    .children
                    .wrap(ManuallyDrop::into_inner(node.children.leaf))
            })
        }
    }

    /// Returns the node from the container without dropping it. The returned
    /// node can only be wrapped again with the same level and [`Alloc`] that
    /// were used to create the container.
    pub(crate) fn leak(self) -> Node<B, Key, Value> {
        // SAFETY: self.node is not read after being moved out of the
        // container - self is dropped immediately afterwards.
        let node = unsafe { ptr::read(&self.node) };
        mem::forget(self);
        node
    }
}

impl<B, Key, Value> Drop for NodeContainer<B, Key, Value> {
    fn drop(&mut self) {
        // SAFETY: NodeContainer has exclusive ownership of self.node, so it is
        // safe to drop.
        unsafe { self.borrow_mut().drop() }
    }
}

impl<B, Key, Value> Clone for NodeContainer<B, Key, Value>
where
    B: Clone,
    Key: Clone,
    Value: Clone,
{
    fn clone(&self) -> Self {
        unsafe { self.alloc.clone_node(self.borrow()) }
    }
}

impl<B, Key, Value> Debug for NodeContainer<B, Key, Value>
where
    B: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("bounds", &self.node.bounds)
            .field("node", &self.borrow().children())
            .finish()
    }
}

pub(crate) struct NodeRef<'a, S, Key, Value> {
    level: usize,
    node: &'a Node<S, Key, Value>,
}

impl<'a, B, Key, Value> NodeRef<'a, B, Key, Value> {
    /// # Safety
    ///
    /// The provided `level` must match the implicit level of the node. It is
    /// undefined behavior to use the returned reference if this condition is
    /// violated.
    pub(crate) unsafe fn new(node: &'a Node<B, Key, Value>, level: usize) -> Self {
        NodeRef { level, node }
    }

    pub(crate) fn children(&self) -> NodeChildrenRef<'a, B, Key, Value> {
        if let Some(level) = NonZeroUsize::new(self.level) {
            NodeChildrenRef::Inner(InnerNodeChildrenRef {
                level,
                // SAFETY: The node is an inner node, so the children are
                // initialized as inner children.
                children: unsafe { &self.node.children.inner },
            })
        } else {
            // SAFETY: The node is a leaf node, so the children are initialized
            // as leaf children. `self.alloc.children` was used to create the
            // node's children.
            NodeChildrenRef::Leaf(unsafe { &self.node.children.leaf })
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
        B: Contains<B>,
        Key: Borrow<Q>,
        Q: Eq + Bounded<B> + ?Sized,
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

    pub(crate) fn _debug_assert_bvh(&self) -> B
    where
        B: Bounds + Debug + Eq,
        Key: Bounded<B>,
    {
        let bounds = match self.children() {
            NodeChildrenRef::Inner(children) => {
                B::union_all(children.iter().map(|child| child._debug_assert_bvh()))
            }
            NodeChildrenRef::Leaf(children) => {
                B::union_all(children.iter().map(|(key, _)| key.bounds()))
            }
        };

        assert_eq!(self.node.bounds, bounds);
        bounds
    }

    pub(crate) fn _debug_assert_eq(&self, other: &NodeRef<B, Key, Value>)
    where
        B: Debug + Eq,
        Key: Debug + Eq,
        Value: Debug + Eq,
    {
        assert_eq!(self.level, other.level);
        self.children()._debug_assert_eq(&other.children());
    }

    pub(crate) fn _debug_assert_min_children(&self, is_root: bool, min_children: usize) {
        let children = self.children();
        if !is_root {
            assert!(children.len() >= min_children);
        }
        if let NodeChildrenRef::Inner(children) = children {
            for child in children {
                child._debug_assert_min_children(false, min_children);
            }
        }
    }
}

impl<'a, B, Key, Value> Debug for NodeRef<'a, B, Key, Value>
where
    B: Debug,
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

impl<'a, B, Key, Value> Bounded<B> for NodeRef<'a, B, Key, Value>
where
    B: Clone,
{
    fn bounds(&self) -> B {
        self.node.bounds()
    }
}

pub(crate) struct InnerNodeChildrenRef<'a, B, Key, Value> {
    level: NonZeroUsize,
    children: &'a FCVec<Node<B, Key, Value>>,
}

impl<'a, B, Key, Value> From<InnerNodeChildrenRefMut<'a, B, Key, Value>>
    for InnerNodeChildrenRef<'a, B, Key, Value>
{
    fn from(children: InnerNodeChildrenRefMut<'a, B, Key, Value>) -> Self {
        InnerNodeChildrenRef {
            level: children.level,
            children: children.children,
        }
    }
}

impl<'a, B, Key, Value> Debug for InnerNodeChildrenRef<'a, B, Key, Value>
where
    B: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, B, Key, Value> InnerNodeChildrenRef<'a, B, Key, Value> {
    fn len(&self) -> usize {
        self.children.len()
    }

    fn iter(&self) -> InnerNodeChildrenIter<'a, B, Key, Value> {
        InnerNodeChildrenIter {
            level: self.level,
            children: self.children.iter(),
        }
    }

    fn _debug_assert_eq(&self, other: &InnerNodeChildrenRef<'a, B, Key, Value>)
    where
        B: Debug + Eq,
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

pub(crate) struct InnerNodeChildrenIter<'a, B, Key, Value> {
    level: NonZeroUsize,
    children: slice::Iter<'a, Node<B, Key, Value>>,
}

impl<'a, B, Key, Value> IntoIterator for InnerNodeChildrenRef<'a, B, Key, Value> {
    type Item = NodeRef<'a, B, Key, Value>;
    type IntoIter = InnerNodeChildrenIter<'a, B, Key, Value>;

    fn into_iter(self) -> Self::IntoIter {
        InnerNodeChildrenIter {
            level: self.level,
            children: self.children.iter(),
        }
    }
}

impl<'a, B, Key, Value> Iterator for InnerNodeChildrenIter<'a, B, Key, Value> {
    type Item = NodeRef<'a, B, Key, Value>;

    fn next(&mut self) -> Option<Self::Item> {
        self.children
            .next()
            // SAFETY: The level of a child of an inner node is always one less
            // than the level of the inner node. The children of an inner node
            // are allocated by the same Alloc as the inner node.
            .map(|node| NodeRef {
                node,
                level: self.level.get() - 1,
            })
    }
}

pub(crate) enum NodeChildrenRef<'a, B, Key, Value> {
    Inner(InnerNodeChildrenRef<'a, B, Key, Value>),
    Leaf(&'a [(Key, Value)]),
}

impl<'a, B, Key, Value> Debug for NodeChildrenRef<'a, B, Key, Value>
where
    B: Debug,
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

impl<'a, B, Key, Value> NodeChildrenRef<'a, B, Key, Value> {
    fn len(&self) -> usize {
        match self {
            NodeChildrenRef::Inner(children) => children.len(),
            NodeChildrenRef::Leaf(children) => children.len(),
        }
    }

    fn _debug_assert_eq(&self, other: &NodeChildrenRef<B, Key, Value>)
    where
        B: Debug + Eq,
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

struct InnerNodeChildrenRefMut<'a, B, Key, Value> {
    alloc: Alloc,
    level: NonZeroUsize,
    children: &'a mut FCVec<Node<B, Key, Value>>,
}

impl<'a, B, Key, Value> InnerNodeChildrenRefMut<'a, B, Key, Value> {
    fn children_mut(&mut self) -> FCVecRefMut<Node<B, Key, Value>> {
        // SAFETY: The level of a child of an inner node is always one less than
        // the level of the inner node. The children of an inner node are
        // allocated by the same fc_vec::Alloc as the inner node.
        unsafe { self.alloc.children.wrap_ref_mut(self.children) }
    }

    /// Drops the children of the node. After calling this method, the children
    /// and references to them are invalid and must not be used.
    ///
    /// # Safety
    ///
    /// It is undefined behavior to drop children of a node that have already
    /// been dropped.
    ///
    /// It is undefined behavior to use children of a node after dropping them.
    unsafe fn drop(mut self) {
        for child in self.iter_mut() {
            // SAFETY: It is undefined behavior if the child has already been
            // dropped.
            child.drop();
        }
        // SAFETY: It is undefined behavior if the children vector has already
        // been dropped.
        self.children_mut().drop();
    }

    fn len(&self) -> usize {
        self.children.len()
    }

    fn at_mut<'b>(&'b mut self, index: usize) -> NodeRefMut<'b, B, Key, Value> {
        // SAFETY: The level of a child of an inner node is always one less than
        // the level of the inner node. The children of an inner node are
        // allocated by the same Alloc as the inner node.
        unsafe {
            self.alloc
                .wrap_ref_mut(&mut self.children[index], self.level.get() - 1)
        }
    }

    fn swap_remove(&mut self, index: usize) -> NodeContainer<B, Key, Value> {
        let child = self.children_mut().swap_remove(index);
        // SAFETY: The level of a child of an inner node is always one less than
        // the level of the inner node. The children of an inner node are
        // allocated by the same Alloc as the inner node.
        unsafe { self.alloc.wrap(child, self.level.get() - 1) }
    }

    fn iter<'b>(&'b self) -> InnerNodeChildrenIter<'b, B, Key, Value> {
        InnerNodeChildrenIter {
            level: self.level,
            children: self.children.iter(),
        }
    }

    fn iter_mut<'b>(&'b mut self) -> InnerNodeChildrenIterMut<'b, B, Key, Value> {
        InnerNodeChildrenIterMut {
            alloc: self.alloc,
            level: self.level,
            children: self.children.iter_mut(),
        }
    }

    fn try_push(
        &mut self,
        node: NodeContainer<B, Key, Value>,
    ) -> Option<NodeContainer<B, Key, Value>> {
        if node.level != self.level.get() - 1 {
            panic!("Cannot push a node with the wrong level");
        }
        self.children_mut()
            .try_push(node.leak())
            // SAFETY: The level of a child of an inner node is always one less
            // than the level of the inner node. The children of an inner node
            // are allocated by the same Alloc as the inner node.
            .map(|node| unsafe { self.alloc.wrap(node, self.level.get() - 1) })
    }

    fn split(
        &mut self,
        splitter: &mut impl Splitter<B>,
        overflow_node: NodeContainer<B, Key, Value>,
    ) -> (B, NodeContainer<B, Key, Value>)
    where
        B: Clone,
    {
        if overflow_node.level != self.level.get() - 1 {
            panic!(
                "cannot split with overflow node of level {} when inner node has level {}",
                overflow_node.level,
                self.level.get()
            );
        }
        let (new_bounds, sibling_bounds, sibling_children) = splitter.split(
            self.alloc.min_children,
            self.children_mut(),
            overflow_node.leak(),
        );
        // SAFETY: sibling_children contains children of the same level as
        // self.children and overflow_node, at self.level - 1. They are
        // allocated by the same Alloc as self.children. The level of the
        // sibling node is the same as the level of the current node.
        (new_bounds, unsafe {
            self.alloc.wrap(
                Node::new(
                    sibling_bounds,
                    NodeChildren {
                        inner: ManuallyDrop::new(sibling_children.leak()),
                    },
                    self.level.get(),
                ),
                self.level.get(),
            )
        })
    }
}

struct InnerNodeChildrenIterMut<'a, B, Key, Value> {
    alloc: Alloc,
    level: NonZeroUsize,
    children: slice::IterMut<'a, Node<B, Key, Value>>,
}

impl<'a, B, Key, Value> IntoIterator for InnerNodeChildrenRefMut<'a, B, Key, Value> {
    type Item = NodeRefMut<'a, B, Key, Value>;
    type IntoIter = InnerNodeChildrenIterMut<'a, B, Key, Value>;

    fn into_iter(self) -> Self::IntoIter {
        InnerNodeChildrenIterMut {
            alloc: self.alloc,
            level: self.level,
            children: self.children.iter_mut(),
        }
    }
}

impl<'a, B, Key, Value> Iterator for InnerNodeChildrenIterMut<'a, B, Key, Value> {
    type Item = NodeRefMut<'a, B, Key, Value>;

    fn next(&mut self) -> Option<Self::Item> {
        self.children
            .next()
            // SAFETY: The level of a child of an inner node is always one less
            // than the level of the inner node. The children of an inner node
            // are allocated by the same Alloc as the inner node.
            .map(|node| unsafe { self.alloc.wrap_ref_mut(node, self.level.get() - 1) })
    }
}

enum NodeChildrenRefMut<'a, B, Key, Value> {
    Inner(InnerNodeChildrenRefMut<'a, B, Key, Value>),
    Leaf(FCVecRefMut<'a, (Key, Value)>),
}

struct InnerNodeChildrenContainer<B, Key, Value> {
    alloc: Alloc,
    level: NonZeroUsize,
    children: FCVec<Node<B, Key, Value>>,
}

impl<B, Key, Value> Debug for InnerNodeChildrenContainer<B, Key, Value>
where
    B: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.borrow().fmt(f)
    }
}

impl<B, Key, Value> IntoIterator for InnerNodeChildrenContainer<B, Key, Value> {
    type Item = NodeContainer<B, Key, Value>;
    type IntoIter = InnerNodeChildrenIntoIter<B, Key, Value>;

    fn into_iter(self) -> Self::IntoIter {
        let alloc = self.alloc;
        let level = self.level;
        let children = self.leak();
        InnerNodeChildrenIntoIter {
            alloc,
            level,
            // SAFETY: The level of a child of an inner node is always one less
            // than the level of the inner node. The children vector of an inner
            // node are allocated by the same fc_vec::Alloc as the inner node.
            children: unsafe { alloc.children.wrap(children) }.into_iter(),
        }
    }
}

impl<'a, B, Key, Value> Drop for InnerNodeChildrenContainer<B, Key, Value> {
    fn drop(&mut self) {
        // SAFETY: InnerNodeChildrenContainer has exclusive ownership of
        // self.children, so it is safe to drop.
        unsafe { self.borrow_mut().drop() }
    }
}

impl<'a, B, Key, Value> InnerNodeChildrenContainer<B, Key, Value> {
    fn borrow(&'a self) -> InnerNodeChildrenRef<'a, B, Key, Value> {
        InnerNodeChildrenRef {
            level: self.level,
            children: &self.children,
        }
    }

    fn borrow_mut(&'a mut self) -> InnerNodeChildrenRefMut<'a, B, Key, Value> {
        InnerNodeChildrenRefMut {
            alloc: self.alloc,
            level: self.level,
            children: &mut self.children,
        }
    }

    /// Returns the children from the container without dropping them. The
    /// returned children can only be wrapped again with the same level and
    /// [`Alloc`] that were used to create the container, and must be dropped
    /// to avoid memory leaks.
    fn leak(self) -> FCVec<Node<B, Key, Value>> {
        // SAFETY: self.children is not read after being moved out of the
        // container - self is dropped immediately afterwards.
        let children = unsafe { ptr::read(&self.children) };
        mem::forget(self);
        children
    }
}

struct InnerNodeChildrenIntoIter<B, Key, Value> {
    alloc: Alloc,
    level: NonZeroUsize,
    children: fc_vec::IntoIter<Node<B, Key, Value>>,
}

impl<B, Key, Value> Iterator for InnerNodeChildrenIntoIter<B, Key, Value> {
    type Item = NodeContainer<B, Key, Value>;

    fn next(&mut self) -> Option<Self::Item> {
        self.children
            .next()
            // SAFETY: The level of a child of an inner node is always one less
            // than the level of the inner node. The children of an inner node
            // are allocated by the same Alloc as the inner node.
            .map(|node| unsafe { self.alloc.wrap(node, self.level.get() - 1) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.children.size_hint()
    }
}

impl<B, Key, Value> Drop for InnerNodeChildrenIntoIter<B, Key, Value> {
    fn drop(&mut self) {
        for _ in &mut *self {}
    }
}

enum NodeChildrenContainer<B, Key, Value> {
    Inner(InnerNodeChildrenContainer<B, Key, Value>),
    Leaf(FCVecContainer<(Key, Value)>),
}

impl<B, Key, Value> Debug for NodeChildrenContainer<B, Key, Value>
where
    B: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.borrow().fmt(f)
    }
}

impl<B, Key, Value> NodeChildrenContainer<B, Key, Value> {
    fn level(&self) -> usize {
        match self {
            NodeChildrenContainer::Inner(children) => children.level.get(),
            NodeChildrenContainer::Leaf(_) => 0,
        }
    }

    fn borrow(&self) -> NodeChildrenRef<B, Key, Value> {
        match self {
            NodeChildrenContainer::Inner(children) => NodeChildrenRef::Inner(children.borrow()),
            NodeChildrenContainer::Leaf(children) => NodeChildrenRef::Leaf(children.borrow()),
        }
    }

    /// Returns the children from the container without dropping them. The
    /// returned children can only be wrapped again with the same level and
    /// [`Alloc`] that were used to create the container, and must be dropped
    /// to avoid memory leaks.
    fn leak(self) -> NodeChildren<B, Key, Value> {
        match self {
            NodeChildrenContainer::Inner(children) => NodeChildren {
                inner: ManuallyDrop::new(children.leak()),
            },
            NodeChildrenContainer::Leaf(children) => NodeChildren {
                leaf: ManuallyDrop::new(children.leak()),
            },
        }
    }
}
