# dLux implementation style

Aim for code a new postgraduate student can read, explain and modify. Mathematical
clarity matters more than minimum line count or hiding every repeated assignment.

## Show the calculation

Lay non-trivial methods out in short logical blocks: prepare inputs, calculate, package
the result. Comments should state each block's purpose, not narrate every line. Explain
non-obvious axis changes and physical conventions near the operation that needs them.
Use a short docstring for advanced mechanisms rather than a long trail of inline
comments. Simple code does not need a comment on every line.

Prefer:

```python
basis, coeffs = zdx.resolve((self.basis, self.coeffs), **context)

# Contract the trailing coefficient axis with the basis vectors.
output = np.tensordot(coeffs, basis, axes=((-1,), (0,)))
return output
```

Avoid combining resolution, conversion and calculation repeatedly in nested calls.
Give intermediate values useful names when they explain a stage. Do not extract a
one-use helper solely to shorten its caller, or introduce generic assignment helpers
for normal constructor fields.

Keep related lines visually balanced. Calls and conditions that fit comfortably on
one line should stay there; do not add trailing commas that force short calls into
vertical lists. When an expression is too long, prefer a meaningful intermediate
value or an ordinary if/else over a vertically split conditional expression. Keep
multiline layout when it reveals structure, such as a matrix or a substantial argument
list. Add blank lines between preparation, calculation and packaging blocks, not
between every statement. A formatter passing is not evidence of readable code.

Use short local names where the meaning is already clear, such as ndim within a
shape calculation. Import jax.tree as jtu and use jtu.map/leaves/flatten for tree
operations. Ordinary loops are preferable when comprehensions hide control flow.

## Classes and constructors

Order a concrete class as fields, constructor, derived properties, primary operations,
secondary methods and private implementation details. Public extension bases may start
with the abstract method defining their contract.

Constructors should convert, check useful shapes and assign ordinary fields:

```python
self.width = zdx.as_array(width, dtype=float)
self.angles = zdx.as_array(angles, dtype=float)
self.invert = bool(invert)
self.alias = alias
```

Store Expressions unchanged. Check array shapes only when arrays are supplied; do not
evaluate an Expression in a constructor to discover its eventual shape. Declare
ordinary defaults such as `invert: bool = False` on the class so Equinox printing
naturally omits them. Do not add pdoc configuration, custom reprs or hidden storage to
make an unnecessarily complex object appear simple. For an option whose value is
useful to inspect (notably interpolation `method`), keep the constructor default but
omit a matching class field default so it remains printed. Do not mark it static
merely to affect its representation.

List inherited public attributes alongside new fields where this clarifies the class
contract (for example Basis.unit), preserving the parent's type and field metadata.
Do not redeclare an abstract property as an unimplemented ordinary annotation.

Use Module directly unless a domain base class adds shared behaviour or an extension
contract. Avoid empty marker subclasses, aliases for renamed unreleased classes, and
inheriting an unrelated active parameter merely to reuse a few methods. Constructor
`**kwargs` forwarding is not a universal pattern: explicit parameters make contracts
and examples easier to follow.

Public class docstrings explain their concept and physical contract. Every concrete
constructor documents its arguments in a NumPy-style Parameters section, including
units, shapes, defaults and meaningful construction effects. Main user classes include
a complete, executable Examples block. Public methods document usable input/output
contracts; simple properties and exact wrappers can remain concise. Follow the
[documentation standard](../dlux-documentation/content.md), and preserve useful examples
when implementations change.

## Modules and reuse

Place module functions after imports/exports and before classes. Keep functions with
the calculation they support. A reusable numerical operation may deserve a public function; an operation whose
meaning belongs to one class may belong on that class. Use a private helper for a real
implementation detail, not merely because it is currently called once. Split files by
useful concepts, not simply because a helper is private. Avoid scattering one operation through base/helper/validation files.

Search existing utilities before writing interpolation, Gaussian, basis, padding,
coordinate, propagation or normalisation mathematics. Core normally uses
`import dLux.utils as dlu`; utility siblings can use direct relative imports. Module
scope imports are preferred. If a cycle appears, review ownership before hiding it in
a function. Deliberate legacy adapters may need an isolated lazy import; do not copy
that pattern into new canonical code.

Use `super()` for normal inheritance. Directly calling a particular base method can be
appropriate when a compatibility override has a different return contract; explain
that exception where it occurs. Do not add wrappers or argument tuples for one call.

Keep canonical implementation separate from compatibility registration. A public
facade should expose intentional names; mutations of another module's exports and
ambiguous name collisions deserve review, not another alias added automatically.

## Review before finishing

Check that each function has an understandable sequence, ownership is visible, physical
and array axes agree, preparation has one clear route, and new abstractions earn their
place. Formatting is a tool; inspect its result instead of assuming formatted code is
readable. Preserve unrelated local edits.
