Unreleased

- Fix `BBITileset.close` raising `RuntimeError: Already borrowed` and losing the reader when a read is in flight. `close` deliberately does not hold the read lock, so it could call `BBIReader.close()` while `pybigtools` held its borrow: that raised, and `close` is what `BaseTileset.__exit__` calls, so an eviction or a `with` exit concurrent with a request failed out of teardown — while the `finally` detached the reader that had *failed* to close, leaving it unreachable and its descriptor to the garbage collector, which an entry below claims to have fixed. Two changes: `FileBacked.close` keeps a reader it could not close, aside rather than republished, so a later `close` reclaims it and `_opened` still reopens; and `BBITileset.close` takes the read lock first, so it waits the read out instead of interrupting it. Measured on a factory-backed tileset costing 10 ms a read, closing 50 ms into a request, 5 of 5 trials: before, `close` raised and the request died with `read of closed file`; after, `close` returns and the request is served, for a wait of 37–46 ms. Under ordinary page-cached load, 8 threads, the wait went from a median of 0.11 ms to 0.15 ms and the requests that saw a closed file fell from 49 to 8 across 12 trials
- Read a BBI file's header inside the open, by implementing `_validate`, so a file that `pybigtools.open` accepts but whose header does not parse is refused with nothing published. BBI was the one backend leaving the base's hook as a no-op, so it had no refusal at that seam at all and a bad file was cached as a reader whose every later use failed differently from its first
- Release a tileset's handle and reader outside the lock that guards opening, with only the detaching done under it. For a factory source the handle is the caller's remote connection and closing it is a round trip, so holding the lock across it stalled every concurrent first-access on a tileset that was already detached — measured at a 355 ms wait on a reopen behind a handle whose `close` takes 400 ms, all of it lock-wait. Nothing needed the lock held that long: a concurrent open calls the factory for a *fresh* handle
- Correct `BBITileset.reading`'s recorded contract, which licensed a usage that corrupts tiles. It said the eagerness of the two read helpers was incidental and that a helper handing back a lazy iterator would be safe — so the obvious next optimization, streaming records rather than accumulating them, was documented as sound. It is not: `records()` returns a lazy iterator and *advancing* it is a read, so a drain that outlives its `reading()` block reads through a borrow the block no longer holds. Measured on a handle-backed reader, five threads x twenty scans: 76 of 100 died with `pyo3_runtime.PanicException ... BadData`, and all 24 that returned returned truncated lists — 72, 172, 196 and 372 records against a truth of 900, which raises nothing and reaches a client as a well-formed wrong tile. Draining inside the block, 0 of 100 and every count exact. The invariant is now stated as the invariant: every read a block starts must finish inside it
- Drop "through one handle" from the borrow refusal a client sees. The borrow is on the reader, not the handle, so the message named a cause that does not apply to a path-backed source
- Narrow `Source.open`'s published return type to a new `BinaryHandle` protocol — `read`, `seek`, `tell`, `close` — which is exactly what the handle check enforces. Annotated `IO[bytes]`, it promised a caller two dozen members that a conforming handle is never asked for and a minimal one does not have, so `with source.open() as h:` type-checked and raised. Narrowed rather than widening the check, because both readers this package uses drive a minimal handle happily and widening would have refused a source the library demonstrably serves. `Source.open_text` is the one method that needs more than the four, since `io.TextIOWrapper` asks for `readable`/`writable`/`seekable`, and its docstring now says so
- Declare `CoolerTileset.block_reader_cls` as `type[BlockReader]` and state what a substitute owes. It was typed bare `type` with no contract, while the batch loop constructs it with `(clr, canvas, balance)`, enters it with `with`, and calls `prefetch`, `block` and `close` — five things a from-scratch substitute had to guess at, and the repo's own test contradicted the declared type by assigning a plain function
- Make `FileBacked`'s lazy reader accessor private, as `_opened()`, leaving `reading()` as the only route to a reader. The two were both reachable and only one was safe: `pybigtools` holds a borrow on its reader for the length of a query, so two threads driving the bare reader of a factory-backed BBI tileset raised `RuntimeError: Already borrowed` — measured at 280 errors across 8 threads where `reading()` produced none — and `RuntimeError` is not a `TileError`, so it escaped the per-tile boundary and failed the whole batch. The unsafe route was also the shorter name and the one a caller reaches for first. Renamed rather than merely hidden, because the member returns a `pybigtools.BBIReader` as often as an `h5py.File`, so `file` misdescribed it half the time. Not marked breaking: `clodius/tiles_v2/__init__.py` is empty and both classes live under leading-underscore modules, so there was no public name to break — and note what the demotion does not buy, since the original entry overstated it: `reading()` yields the reader and nothing invalidates it on exit, so a caller who keeps what it yielded can still drive it outside the block. The constraint is stated, not enforced
- [BREAKING] Rename `CoolerTileset.reader` to `block_reader_cls` and document it. It was an undocumented public attribute holding the block-reader *class* selected by `batched`, so the old name said "reader" for something that is not one, beside an inherited `_reader` that is. It stays public because substituting it is the only way to observe how the batch loop opens, and a private name would have pushed callers — and this repo's own tests — into private state
- Make the BBI read lock reentrant. `reading()` is published as the way to reach the reader, so a caller may nest it — hold the reader across a helper, compare two ranges — and a non-reentrant lock turned that into a permanent hang with no exception and no timeout. Only for a factory-backed source: the path route skips the lock, so a path-backed test could never have seen it
- Resolve the reader before taking the BBI read lock, rather than inside it, so the critical section covers the read and nothing else. Scope hygiene rather than a saving: `_opened()` takes its own lock and double-checks, so exactly one thread performs the open and the rest block regardless — a cold eight-thread first batch against a factory sleeping 50–100 ms per open measured the same either way, crossing in both directions
- Skip the BBI read lock for a genomic range that is past the end of the genome. Such a range returns a NaN pad without touching the reader, so taking the lock serialized other threads behind no I/O at all — measured at 554 of 2075 lock entries on a small fixture, since a quadtree pads to a power of two. The pad itself is still produced and still counted: grid reconciliation is given the expected bin count and refuses a short grid, and the padding is what fills a low-zoom tile's trailing cells
- Release what a tileset's constructor opened without being able to fail the constructor. The release is an optimization rather than a correctness requirement, and `close` detaches the handle before closing it, so a handle whose `close` raised — a remote connection torn down badly — failed a tileset that had otherwise constructed, and left the handle unreachable
- Require `close` of a factory's handle, alongside `read`, `seek` and `tell`. The tileset closes what it opened, so a handle without one passed validation and then failed from inside `close` as a bare `AttributeError`, with the handle already detached — and now that construction releases its header read, that surfaced at construction
- Refuse a source callable that takes arguments at the `Source` constructor, not only at `Source.coerce`. `Source` is a member of `SourceLike` and `coerce` passes an already-normalized one straight back out, so a guard living only in `coerce` was bypassed by the documented shape and the failure resurfaced frames later as `open() missing required argument 'file'` — exactly what the guard exists to prevent
- Remove `Source.for_reader`. It returned oxbow's `str | Callable` union, and the extraction of the shared file-backed protocol removed its last caller: neither `h5py` nor `pybigtools` accepts the callable arm, so both read `Source.path` instead. It is a two-line expression to reconstruct in the slice that needs it
- Hand a path-backed BBI file to `pybigtools` unbuffered. Before the shared protocol landed, the BBI route opened its own handle with Python's default buffering. This affects only a path whose suffix `pybigtools` does not dispatch on — which is the *only* route in the package that opens a handle from a path at all, since `_needs_handle()` is false for every path-backed HDF5 tileset — and it measured at 1.00–1.02x either way, with identical read counts and identical bytes, so it is recorded as a behaviour change rather than a performance one
- Serialize a BBI read through `BBITileset.reading`, a context manager that is also the seam a caller reaching past `tiles` should use. The reader it yields carries a constraint the tileset owns — `pybigtools` holds a borrow on a handle for the length of a query — and a caller driving the reader directly previously had no way to honour it, because the lock was private and attached to `tiles`
- [BREAKING] Make `BBIInteractionLinksTileset.link_policy` a read-only property, matching the other two link tilesets. A bare public attribute let a value assigned after construction skip the `LinkPolicy` coercion, and the tile loop's catch-all arm then served every interaction whose hull crossed the tile under the id of whatever was asked for, with nothing raised
- Refuse a source callable that takes arguments, naming the fix. `Source.coerce` tested only that an object was callable, so `builtins.open` was accepted and failed several frames later as "open() missing required argument 'file'" — a message naming neither the tileset nor the repair, which is exactly what that method exists to prevent
- Refuse a handle that cannot seek, or that yields `str`, by testing the capability alongside its declaration. The checks looked only for a declared `seekable()` and for `io.TextIOBase` membership, so a duck-typed handle that declared neither was admitted and failed inside the reader; probing the capability catches that. The declaration is still consulted for seek, because a handle whose `seekable()` reports `False` while its `seek` happens to succeed is one a reader that asks before seeking will believe — so it is refused rather than admitted on the probe alone
- Serialize reads through a BBI reader this tileset opened the handle for. `pybigtools` holds a borrow on that Python handle for the duration of a query, so two threads reading through one handle-backed reader raised `RuntimeError: Already borrowed` — and since `RuntimeError` is not a `TileError` it escaped the per-tile boundary and failed the whole batch as a 500. A factory-backed tileset therefore could not serve concurrent requests, which is the only way a tile server serves them: measured at 200 of 400 requests served and 4 of 8 threads dead. The ceiling that leaves is real and is recorded in `BBITileset`'s docstring: one tileset holds one reader, so a factory-backed tileset's reads do not scale with a worker pool — flat 177–213 queries/s and 632–648 tiles/s from 1 to 8 threads, with 4.2–7.8x recoverable by giving each thread its own reader. A bounded reader pool is filed against the oxbow epic; the lock is the correct minimal fix until then, since removing it refuses tiles rather than slowing them. A path-backed reader is exempt, and the reason recorded here first was wrong: the borrow is on the `BBIReader` object — a PyO3 `RefCell` — not on the Python handle, so there is a borrow to contend on either route, and the same workload against a path-backed reader in an interpreter where `numpy` has not been imported refuses 2800 of 3200 queries. What was measured instead is that the overlap does not occur in this process: with `numpy` imported, a path-backed query does not hand the interpreter back until it returns, and the same workload refuses 0 of 3200. The exemption therefore rests on a precondition of the process that nothing enforces, which is now recorded at the import it depends on. The two routes cost the same per query; what the path route does not pay is the lock. The lock is held around a single reader call rather than a whole batch: the borrow lasts one query, and a batch is mostly work that never touches the reader, so a batch-wide lock put every concurrent request behind all of it — a trivial tile served alongside two batch threads measured 833 ms at p50 and 7.5 ms once narrowed. A residual borrow error is translated into a `TileError` so it reaches a client as that tile's refusal, and that translation is now written once rather than twice with two different catch widths and a hardcoded file flavour that was wrong for half the files reaching it
- Fix `BBIInteractionLinksTileset` orphaning the reader and handle its base class opened when its own `link_policy` argument was invalid. It was the one BBI tileset doing work after the guarded `super().__init__` returned, so the guard could not reach it; the policy is now coerced before anything opens, which removes the window rather than cleaning up after it. For a factory-backed source the orphan was the caller's remote connection, whose `close()` was never called
- Move the constructor-release guard into `FileBacked` as a `_configuring()` context manager each subclass enters, replacing three hand-written copies of `try/except Exception: self.close(); raise` whose rationale comment was byte-identical between `cooler.py` and `bbi.py`. It releases on a normal return as well, which retired a second three-way copy — a trailing `self.close()` under another byte-identical comment — that the first extraction introduced in the same pass. It was the one element of the open-and-release protocol the extraction left behind, and the first subclass to do work outside a copy of it leaked immediately. The guard also widens to `BaseException`, so a `KeyboardInterrupt` landing between the open and the end of construction no longer orphans what was opened, and it suppresses a failing `close` during teardown rather than letting it replace the construction error the caller needed — a server renders a `TilesetUnavailable` as a refusal and an `OSError` as a 500, so which one escapes decides what the operator sees. A context manager rather than a hook the base calls: a zero-argument hook made every subclass copy its constructor arguments onto `self` to read them back one statement later, and the requirement to do that *before* `super().__init__` was written down nowhere
- Release what a tileset's constructor opened to read its header, in all three file-backed backends, so registering one holds nothing. Construction still *reads* the header — a source that is not this tileset's format is still refused at registration rather than at the first request — it simply no longer keeps what it opened. A server registers tilesets it may never serve, and for a factory source each one it held was a live remote connection — 200 idle registrations held 200 descriptors. A served tileset now pays one extra open and an unserved one pays none. `CoolerTileset` and `MultivecTileset` held one connection per registered dataset until this; all three backends now answer the same way
- Admit `Source` to the `SourceLike` union. Every tileset constructor already accepted one at runtime — `Source.coerce` passes an already-normalized `Source` straight back out, and `Source.sibling` and `Source.optional` both return one — so a caller who normalized once and reused the result was writing code the annotation rejected and the runtime accepted
- Fix a bigWig or bigBed whose suffix is spelled in a case `pybigtools` does not dispatch on — `signal.BW`, `track.BigWig` — failing to open at all. The suffix check case-folded before comparing, so it claimed every case variant was dispatchable and handed the path string to `pybigtools.open`, which refuses anything but the six exact spellings `.bw`, `.bigwig`, `.bigWig`, `.bb`, `.bigbed`, `.bigBed` with "Invalid file type". The handle route opens every one of these files, so the fast path was turning an openable file into an unopenable tileset; the check now matches what `pybigtools` accepts. Note the asymmetry the fix exposes: *failing* to recognize a suffix costs one slower open, because the handle route opens any BBI file whatever it is named, but *recognizing* one commits to the flavour it names — `pybigtools.open(<path>)` picks the parser from the extension while `pybigtools.open(<handle>)` sniffs the bytes, so a bigWig named `.bb` is refused by the path branch and served by the handle branch. Being unrecognized is strictly safer than being mis-recognized
- Document the `source` parameter on the BBI subclasses, and name it in `BBIInteractionLinksTileset`'s signature. A subclass that defines any docstring stops inheriting the base's, so `help(BBISignalTileset)` showed a widened first parameter with nothing saying a callable was accepted; and the one subclass with its own keyword swept `source` into `*args`, hiding it from `inspect.signature` as well. That subclass now spells out all five shared parameters rather than only `source`, so every BBI tileset publishes the same readable signature instead of one of them publishing a caller `*args` where its siblings publish five named parameters
- Let the five BBI tilesets take a file-like source in place of a path, so a bigWig or bigBed can be served from remote storage without first writing it to local disk. The `.bw`/`.bigwig`/`.bb`/`.bigbed` fast path is preserved — those suffixes still reach `pybigtools.open` as the path string, which is the same call a reader that had never heard of `Source` would make — and every other source is handed a live handle, the branch that already existed for a bigInteract but that no caller could reach
- The HDF5 and BBI tilesets now share one open-and-release protocol `H5Backed` and `BBITileset` had implemented twice by hand. `H5Backed` is now three lines over it and `BBITileset` inherits the fixes that went into the h5py copy — the lazy open with no lock, so concurrent first access called the caller's factory N times and left N-1 handles `close` could not reach; the `close` that stranded the handle whenever the reader's close raised; and the publication order a concurrent `close` could slip between — rather than carrying its own copy of all three
- [BREAKING] Rename the BBI tilesets' first parameter from `path` to `source`, and annotate it `SourceLike`, matching `CoolerTileset` and `MultivecTileset`. The parameter accepts a path or a factory, so `path` named something it no longer is; every construction in the tree passes it positionally, so nothing moves
- Fix `BBITileset.close` never closing its reader. It dropped the reference and left the `pybigtools.BBIReader` for the garbage collector, so a path-backed tileset released its file whenever the collector got to it rather than when it was told to; `BBIReader.close()` does not close a Python handle it was given either, so releasing a tileset means closing both, reader first, because the reader reads from the handle lazily
- Fix the BBI tilesets orphaning what they opened when construction failed *after* the open succeeded — a file that opens as a BBI but whose chromsizes do not survive quadtree validation. The constructor raises, so the caller never receives an object to call `close` on; for a factory-backed source the orphan is the caller's remote connection
- [BREAKING] Refuse a cooler with no `resolutions` group with `TilesetUnavailable` rather than `ValueError`. The file is readable and simply is not this tileset's format, which is a whole-request fault a server should render as a readable refusal; a bare `ValueError` is not a `TilesetError`, so it escaped the per-tile boundary as a 500 — the same outcome the cached-`KeyError` bug produced, which is what making the refusal repeatable was meant to fix
- Make `Source`'s constructor keyword-only and normalizing. It became public when the module moved to `clodius.core`, and it had held none of the invariants `Source.coerce` establishes: a factory passed positionally landed in the path slot and surfaced as a `__str__` that raises, from inside the error-reporting path that exists to name the file. A non-callable factory is now refused there too, not only by `coerce`
- Route `variant`'s sibling-index detection through `Source.has_sibling`, extending the reconciliation the `bed`/`bedpe`/`gxf` migration began. `alignment.py` still derives its sibling index with its own `os.path.exists` loop and is left to its own slice, so the tree is not yet down to one rule. `VariantFormat.has_sibling_index` still tested `os.path.exists`, so a directory named `records.vcf.gz.tbi` made a VCF declare itself indexed where `Source` said otherwise
- Fix the lazy reader accessor (then `H5Backed.file`, now the private `FileBacked._opened`) returning `None` through a signature that promises a reader. The lazy guard read the field twice — once to test, once to return — and only the first read was synchronized, so a concurrent `close` between them handed the caller a `None` that every consumer immediately subscripts
- [BREAKING] Rename `CoolerTileset` and `MultivecTileset`'s first parameter from `path` to `source`, and annotate it `SourceLike`. The parameter accepts a path or a factory, so `path` named something it no longer is; every construction in the tree passes it positionally, so nothing moves. The widened annotation is the public contract the file-like source feature actually ships, and it was missing — a caller reading either constructor had no signal that a callable was accepted
- Hand `h5py` the path string when a tileset is constructed from a path, rather than a handle opened from it. Going through a handle selected HDF5's `fileobj` driver instead of the native `sec2`, which forfeits HDF5's recognition that a file is already open read-only in the process: 50 tilesets over one file cost 50 descriptors where they had cost one, and a few hundred exhausted the default limit. It also degraded `cooler.Cooler.uri` to the repr of a `BufferedReader`, so the uri no longer round-tripped
- Add `clodius.tiles_v2._h5.H5Backed`, holding the open-and-release protocol `CoolerTileset` and `MultivecTileset` had implemented twice by hand, and fix the three defects that lived in one copy or the other: a lazy open with no lock, so concurrent first access called the caller's factory N times and left N-1 handles `close` could not reach; a `close` that stranded the handle whenever the HDF5 close raised, permanently, because `_file` stayed set and every retry failed at the same point; and a publication order in `multivec` that a concurrent `close` could slip between
- Fix both HDF5 tilesets orphaning the handle they opened when construction failed *after* the open succeeded — a file that is valid HDF5 but not the expected format. The constructor raises, so the caller never receives an object to close; for a factory-backed source the orphan is the caller's remote connection
- Refuse a factory handle that is not seekable or not binary, naming the missing capability. The guard tested only for `read` while its own message promised a binary handle, so a `lambda: open(p)` that forgot the mode string reached `h5py` and surfaced as `UnicodeDecodeError`, and a non-seekable handle surfaced as h5py complaining about paths. Neither is a `TileError`, so both reached a client as a 500
- Accept a `bytes` path, and refuse an empty one. `os.fsdecode` already handled bytes and a bytes-yielding `PathLike` was accepted, so admitting only two of the three representations `os` accepts narrowed the parameter at the moment it was being widened. An empty path is truthy as a `Source`, so a consumer's empty-source guard does not fire and it surfaces much later as `FileNotFoundError`
- Route `bed`, `bedpe` and `gxf` sibling-index detection through `Source.has_sibling`, deleting three byte-identical copies of the probe. The copies tested `os.path.exists`, which a *directory* named `<path>.tbi` satisfies, so a tileset declared itself indexed and failed on first open; `Source` tests `os.path.isfile`, and the two disagreeing about the flag that decides range-query-versus-scan was a defect waiting for the first such directory
- Declare `_src: Source` on `BaseTileset`, annotated and unassigned, so the attribute `__repr__` reads is published rather than guessed at
- Fix `CoolerTileset` refusing a file with no `resolutions` group exactly once. The check ran after the opened file was already cached, so the first request got the intended `ValueError` and every later one a bare `KeyError` from the group lookup. Neither is a `TilesetError`, so the refusal reached a client as a 500 either way; the type is corrected in a later entry
- Add `clodius.core.source.Source`, and let `CoolerTileset` and `MultivecTileset` take a file-like source in place of a path, so a caller can serve tiles from remote storage without first writing the file to local disk. The source is a zero-argument callable returning a fresh binary handle — the shape `fsspec`, `smart_open` and a plain `lambda: open(p, "rb")` all satisfy, and the only shape that works for every reader the package uses, since oxbow may reopen its source between queries. An already-open file is refused at construction rather than accepted: the repair a caller reaches for is `lambda: handle`, a factory that returns the same exhausted handle every time, which would serve records from wherever the previous query left the cursor. A path keeps working unchanged — a path-backed source hands the reader back the same string it was given, so that branch is the same call it always was. clodius does not depend on `fsspec`; the contract is stdlib
- Fix `take_most_important` returning every record at a cap of zero. The slice implementing the limit reads `ranked[-cap:]`, which at zero is the whole list, so a server configured to serve no records emitted an unbounded tile. The cap now also accepts `None`, which is what `TilePolicy` supplies when `max_records` is unset
- Fix `TileCanvas.invert` returning a plausible `GenomicRange` for a position past the end of the lattice, and add the construction guards it had lost. Positions past the end of the *genome* but inside the canvas remain in range, since that padding is what fills the trailing NaN bins of a low-zoom tile
- Fix an indexed BED serving its entire contents for a tile lying past the end of the genome. oxbow reads an empty region list as no region filter at all, while the scanning path's pushed-down predicate correctly matched nothing, so compressing a file changed what it served
- Fix `BaseTileset.parse_tile_id` collapsing an empty `options` frozenset to `None`, which means "accept any option" and is the opposite of what an empty set declares
- Fix `TileId.parse` accepting a negative zoom or position, and raising bare `ValueError` for a badly declared arity. Coordinates are now ASCII decimal only: `str.isdigit` admits superscripts that `int` then rejects with a `ValueError` that escapes the server boundary, and `int` itself admits `+5`, `1_0` and fullwidth digits, any of which would denote one tile under several distinct request strings
- Fix `to_tile_record` raising `KeyError` for a record whose contig is absent from the chromsizes, which reached a client through the region listing as a 500
- Fix `from clodius.core import *` raising `AttributeError`: `__all__` listed `Canvas`, a name no submodule binds, and omitted `TileCanvas`, which the facade imports
- Make `TilesetInfo` frozen. Its cached coordinate system is only sound while the field it derives from cannot be reassigned
- Make `stable_importance` pass `usedforsecurity=False`, so the thinning path keeps working on a FIPS-enforcing build where md5 is otherwise refused
- Implement the per-tile error boundary in the eight tilesets that lacked it. `clodius/core/errors.py` documents a `TileError` as caught by `tiles()` and returned as that tile's payload; three tilesets did this and eight raised through the whole batch

v0.22.2

- Fix `tile_functions_parasail` D/I CIGAR inversion: `nw_trace_scan_profile_16`
  returns a CIGAR from the profile's perspective, so insertion and deletion ops
  were swapped relative to SAM convention; normalize before parsing
- Fix `cigar_to_subs` bounds guard: skip X ops that would index past the end of
  the reference (triggered when query sequences are longer than the reference)
- Add insertion and deletion generation to `sequence_pileup.py` benchmark
  (Poisson-sampled ~3 insertions and ~3 deletions per sequence in addition to SNPs)

v0.22.1

- Fix release to actually include latest files

v0.22.0

- Add parasail alignment backend to pileup tile functions (`tile_functions_parasail`,
  `cigar_to_subs`) and expose it via a `method` parameter on `get_pileup_alignment_data`
  (~9× faster than BioPython PairwiseAligner on large batches)
- Fix `get_pileup_alignment_data` tile ID bug (`"0.0"` → `"x.0.0"`) and rename
  return key `"type"` → `"tileset_info"` for consistency
- Make `csv_sequence_tileset_functions` import lazy to avoid pulling in `smart_open`
  on every `import clodius.tiles.pileup`

v0.21.1

- Ability to pass in feature_type to clodius single_tile function

v0.21.0

- Huge set of changes to support file-pointer based tileset functions

v0.20.4

- Fix overflow issue in cooler files

v0.20.3

- Add chromsizes tileset_info function

v0.20.2

- Convert cooler chromsizes to int64 to prevent overflow error with recent versions of h5py and numpy

v0.20.1

- Remove use of deprecated `max_chunk` argument from cooler tile fetcher.

v0.20.0

- remove `numpy` from setup requires
- Use builtin `warnings` module instead of relying on alias from `numpy`.
- Replace instances of `.iteritems()` with `.items()`

v0.19.0

- Fix decoding error in multivec tileset info
- Allow JSON in multivec tileset info

v0.18.1

- Don't pin versions in requirements.txt

v0.18.0

- Added bigwigs_to_multivec command
- Bumped dask version
- Bumped pandas version

v0.17.1

- Fix narrow npmatrix tile fetching bug

v0.17.0

- Updated the BAM file fetcher to do more efficient substitution loading
- Include strand, cigars, and other metadata from reads
- Added tabix loader that can be estimated to estimate the size of data in a region of a BAM file
- Add FASTA tileset

v0.16.0

- Return `chromsizes` as a single array in beddb `tileset_info`
- [BREAKING] Remove the `chrom_names` and `chrom_sizes` fields in the beddb tileset info

v0.15.4

- Remove type=bool from bedpe aggregate function to fix "Got secondary option for non-boolean flag" error
- No default assembly

v0.15.2

- More informative error message when doing bedfile_to_multivec conversion

v0.15.1

- Added support for multivec `row_infos` stored under `/info/row_infos` as an hdf5 utf-8 string dataset.

v0.15.0

- Improve performance of `clodius aggregate bedpe` using sqlite batch inserts, transactions, and PRAGMAs
- Show default values for `clodius aggregate bedpe -h`
- Add short options to `clodius aggregate bedpe`
- Make `clodius aggregate bedpe --chromosome` actually do something
- For bedpedb 1D tiles, retrieve entries where either regions at least partially overlaps with a tile
- Harmonize bedpedb tile getter names

v0.14.3

- Small bug when retrieving tile 0.0 from bam files
- More accurate generation of multivec tiles that span across chromosomes

v0.14.2

- Make sure that chromsizes are serialized as ints

v0.14.1

- Natural ordering of bam file chromosomes

v0.14.0

- Add "name" field to `beddb` format

v0.13.1

- Fix returned header values

v0.13.0

- Ran black on the entire code base
- Introduced versions to beddb and bed2ddb files
- Added zoom level to the r-tree index

v0.12.0

- Added tile queries for bigBed files, using score data (if available) to threshold those elements returned, based on a specified or default maximum.

v0.11.5

- Add tiles function and test for multivec. It can be used in higlass-python and higlass server.

v0.11.4

- Modified bedfile_to_multivec conversion to retain lines in which the end coordinate is not a multiple of the resolution.
  It adds an extra bin for the remainder. It also displays an error when start coordinate is not multiple of resolution
- Fix bug to handle headers.
- Modified create_multivec_multires for states files to show only the contents of the first column of the row_infos file as the name of the state.

v0.11.3

- Maintenance: Switched from nose to pytest and added coverage reporting.
- Use columnar format for BAM reads return values

v0.11.2

- Added `max_tile_width` parameter to ct.bam.tiles() so that users can set a
  limit on how large of a region data is returned for

v0.11.0

- Added bamfile support

v0.10.12

- Simplified density tiles generator.
- Fix error with `npvector` tileset_info.
- Add `__version__`.

v0.10.11

- Calculate np.nan on the fly if not available for npvector tracks

v0.10.10

- Fix error in bedfile_to_multivec conversion when encountering value-less files
- Update the bedpe aggregator to fix the error in using a chromsizes file
- Make tsv_to_mrmatrix more flexible and add it to the exported scripts.
- Display more meaningful error messages when encountering unknown chromosomes or assemblies
- Removed redundant multivec function
- Removed obsolete bigwig function
- Make tsv_to_mrmatrix more flexible and add it to the exported
- Detect non-symmetric square coolers using the storage-mode metadata. Support for the symmetric property is retained for the legacy mcool format.

v0.10.7

- Changed bins_per_dimension in npvector.tileset_info to match the value in
  in npvector.tiles (1024)

v0.10.5

- Removed slugid decode

v0.10.2 (2019-02-06)

- new option to import a "states" file format, a bed file with categorical data, e.g. from chromHMM to multivec format.
- while converting a bed file to multivec, each segment can now be a multiple of base_resolution,
  rather than exactly match the base_resolution

v0.10.1 (2019-01-22)

- Removed a buggy print statement from the conversion script

v0.9.5 (2018-11-11)

- If the start position is greater than the end position, switch them around

v0.9.0 (2018-05-07)

- Removed clodius aggregate bigwig

v0.8.0 (2018-05-06)

- Bug fixes
- GeoJSON aggregation
- Multivec tiles

v0.7.4 (2018-02-14)

- Greatly sped up bedfile aggregation and fixed the maximum per tile limiting

v0.7.3 (2018-01-31)

- Use random importance by default

v0.7.2 (2017-12-04)

- Populate the xEnd field

v0.7.1 (2017-12-04)

- Added the 'header' field to beddb tiles

v0.7.0 (2017-10-04)

- Replaced get_2d_tile with get_2d_tiles
- Replaced get_tile with get_tiles

v0.6.5 (2017-07-14)

- Added a delimiter option to bedfiles
- Fixed beddb uid decoding
- Adding decoding to slugid.nice() methods

v0.6.0-0.6.4 (2017-07-13)

- python3 support

v0.5.0

- Added clodius aggregate bedpe
- More extensive unit tests

v0.4.7

- Renamed tsv to bedlike

v0.4.6

- Bug fixes in tile-text-file (tsv tiling)

v0.4.3

- When extracting a certain chromosome, place features at the position of the
  chromosome
