# Contributing to CuWeaver

Thank you for considering a contribution to **CuWeaver**, a CUDA concurrency library.  
Contributions are welcome from anyone. This guide explains how to propose changes, report
bugs, and adhere to project conventions.

## How to Contribute

1. **Fork the repository** on GitHub and clone your fork locally.
2. **Create a feature branch** for your changes:
   ```bash
   git checkout -b feature/my-new-feature
   ```

3. **Develop your changes** while following the coding standards described below.  
    Write descriptive commit messages and keep commits focused.
    
4. **Write or update tests.** All changes must be accompanied by tests that verify  
    the intended behaviour and protect against regressions.
    
5. **Document your changes.** Public APIs, classes, functions and namespaces must  
    include Doxygen comments so that documentation can be generated automatically.
    
6. **Ensure the test suite passes**. Run the tests locally (see the  
    [Testing Guide](docs/TESTING.md)) before submitting your changes.
    
7. **Open a pull request** from your feature branch against the `main` branch of  
    the upstream repository. Provide a clear description of the problem solved  
    and reference any related issues. Maintainers will review your pull request  
    and may request changes.
    
8. **Respond to review feedback** in a timely manner. Once the pull request  
    receives approval, it will be merged.
    

## Reporting Bugs

If you encounter unexpected behaviour, please open an issue on GitHub.  
Include the following information to help us reproduce and diagnose the problem:

- A clear description of the bug and how it manifests.
    
- Steps or code snippets to reproduce the issue.
    
- Information about your environment (operating system, CUDA version,  
    compiler version, etc.).
    
- Any relevant log messages or stack traces.
    

## Suggesting Enhancements

We welcome feature requests and enhancement ideas. Before opening an issue,  
please check the existing issues to avoid duplicates. When proposing a new feature:

- Describe the motivation and expected benefit.
    
- Explain how the feature fits within the goals of CuWeaver.
    
- Outline a possible implementation approach.
    

## Coding Standards

To maintain a consistent and readable codebase, CuWeaver adheres to the  
following coding conventions. **These guidelines must be observed for all  
contributions.**

### Namespaces

- All public identifiers reside in the `cuweaver` namespace.  
    Sub‑namespaces may be created for logically grouped functionality but should  
    remain concise and descriptive.
    
- Use only lowercase letters for namespace names.  
    A namespace name should be a single word and not excessively long.


### Naming Conventions

| Element          | Convention                            | Example                        |
|------------------|---------------------------------------|--------------------------------|
| Files            | Descriptive names, free form          | `CudaEvent.cuh`, `helpers.hpp` |
| Classes/Structs  | PascalCase; `cuda` prefix lowercase   | `cudaEvent`, `StreamManager`   |
| Functions        | camelCase (lower camel)               | `recordEvent()`                |
| Variables        | camelCase                             | `deviceCount`                  |
| Macros           | UPPER_CASE with underscores           | `CUW_THROW_IF_ERROR`           |
| Namespaces       | lowercase single word                 | `cuweaver::utils`              |

**Note:** When a class or struct name begins with `cuda`, the prefix `cuda` should remain lowercase (e.g., `cudaStream`). All other class and struct names follow the PascalCase (Upper Camel) convention.

### Commenting and Documentation

CuWeaver uses **Doxygen** to generate API documentation. All public headers,  
classes, functions, templates, enums and namespaces must be documented using  
Doxygen syntax. Use the following patterns:

- **File header**:
    
    ```cpp
    /**
     * @file FileName.cuh
     * @author Your Name (https://yourwebsite.example)
     * @brief Brief description of the file contents.
     *
     * @details Longer description detailing the purpose of the file.
     * Describe major classes or functions declared here.
     */
    ```
    
- **Namespace**:
    
    ```cpp
    /**
     * @namespace cuweaver::utils
     * @brief Brief description of the namespace.
     *
     * @details Longer description of the domain and responsibilities of
     * the namespace, if needed.
     */
    ```
    
- **Class or struct**:
    
    ```cpp
    /**
     * @class CudaEvent
     * @brief Brief description of the class.
     *
     * @details Detailed description explaining design, usage, and any
     * invariants. Omit if not needed.
     *
     * @tparam T Description of template parameter T. // Only when templated
     */
    ```
    
    Member variables should have brief trailing comments using `//!<`:
    
    ```cpp
    int count; //!< Number of items contained in the list.
    ```
    
- **Functions**:
    
    ```cpp
    /*!  
     * @brief Brief description of what the function does.
     *
     * @details Detailed description if necessary. Include algorithm steps,
     * preconditions, side effects or complexity. Separate paragraphs with blank lines.
     *
     * @tparam TypeParam Description of template parameter.        // Only for templates
     * @param[in]  param1 Description of the first parameter.      // Sentences start with a capital letter and end with a period.
     * @param[out] param2 Description of the second parameter.
     * @param[in]  param3 Description of the third parameter.
     * @return Description of the return value.                    // For scalar return values
     * @retval value1 Meaning of return value 1.                   // For multiple discrete return values
     * @retval value2 Meaning of return value 2.
     * @throws ExceptionType Condition under which exception is thrown.   // If applicable
     * @note Additional notes about the function.                   // Optional
     * @warning Warning message if applicable.                      // Optional
     */
    ReturnType functionName(TypeParam param1, TypeParam& param2);
    ```
    
- **Enums**:
    
    ```cpp
    /**
     * @enum ErrorCode
     * @brief Brief description of the enumeration.
     *
     * @details Detailed description if needed.
     */
    enum class ErrorCode { ... };
    ```
    

### Testing Requirements

Every new feature or bug fix **must** include corresponding unit tests.  
If you modify existing functionality, update the relevant tests accordingly  
and ensure that the entire test suite passes. See the [Testing Guide](docs/TESTING.md)  
for instructions on building and executing the tests.

## Commit Messages

- Write clear, concise commit messages that describe _what_ and _why_.
    
- Use the imperative mood (“Add test for stream priority handling”).
    
- Reference related issues or pull requests if applicable (e.g., `Fixes #42`).
    

## Licensing

By contributing to CuWeaver, you agree that your contributions will be  
licensed under the MIT License, the same license that covers the project.