# Defense-in-Depth Validation

> Examples below are language-neutral pseudocode.

## Overview

When you fix a bug caused by invalid data, adding validation at one place feels sufficient. But that single check can be bypassed by different code paths, refactoring, or mocks.

**Core principle:** Validate at EVERY layer data passes through. Make the bug structurally impossible.

## Why Multiple Layers

Single validation: "We fixed the bug"
Multiple layers: "We made the bug impossible"

Different layers catch different cases:
- Entry validation catches most bugs
- Business logic catches edge cases
- Environment guards prevent context-specific dangers
- Debug logging helps when other layers fail

## The Four Layers

### Layer 1: Entry Point Validation
**Purpose:** Reject obviously invalid input at the API boundary

```text
function create_project(name, working_directory):
    if working_directory is empty or blank:
        fail("workingDirectory cannot be empty")
    if not path_exists(working_directory):
        fail("workingDirectory does not exist: " + working_directory)
    if not is_directory(working_directory):
        fail("workingDirectory is not a directory: " + working_directory)
    # ... proceed
```

### Layer 2: Business Logic Validation
**Purpose:** Ensure data makes sense for this operation

```text
function initialize_workspace(project_dir, session_id):
    if project_dir is empty:
        fail("projectDir required for workspace initialization")
    # ... proceed
```

### Layer 3: Environment Guards
**Purpose:** Prevent dangerous operations in specific contexts

```text
function git_init(directory):
    # In tests, refuse git init outside temp directories
    if running_in_test_mode():
        if not normalized(directory) starts_with system_temp_dir():
            fail("Refusing git init outside temp dir during tests: " + directory)
    # ... proceed
```

### Layer 4: Debug Instrumentation
**Purpose:** Capture context for forensics

```text
function git_init(directory):
    log_debug("About to git init", {
        directory: directory,
        cwd: current_working_dir(),
        stack: current_stack_trace(),
    })
    # ... proceed
```

## Applying the Pattern

When you find a bug:

1. **Trace the data flow** - Where does the bad value originate? Where is it used?
2. **Map all checkpoints** - List every point the data passes through
3. **Add validation at each layer** - Entry, business, environment, debug
4. **Test each layer** - Try to bypass layer 1, verify layer 2 catches it

## Example from a Session

Bug: empty `projectDir` caused `git init` to run in the source tree

**Data flow:**
1. Test setup → empty string
2. `Project.create(name, "")`
3. `WorkspaceManager.createWorkspace("")`
4. `git init` runs in the current working directory

**Four layers added:**
- Layer 1: `Project.create()` validates not empty / exists / writable
- Layer 2: `WorkspaceManager` validates projectDir not empty
- Layer 3: `WorktreeManager` refuses git init outside the temp dir in tests
- Layer 4: Stack-trace logging before git init

**Result:** All 1847 tests passed, bug impossible to reproduce

## Key Insight

All four layers were necessary. During testing, each layer caught bugs the others missed:
- Different code paths bypassed entry validation
- Mocks bypassed business logic checks
- Edge cases on different platforms needed environment guards
- Debug logging identified structural misuse

**Don't stop at one validation point.** Add checks at every layer.
