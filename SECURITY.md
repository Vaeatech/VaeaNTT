# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | ✅ Current release |

## Reporting a Vulnerability

If you discover a security vulnerability in VaeaNTT, **please do not open a public issue.**

Instead, report it privately:

1. **Email**: [TODO: security contact email]
2. **Subject**: `[VaeaNTT Security] <brief description>`
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Impact assessment
   - Suggested fix (if any)

We will acknowledge receipt within **48 hours** and provide an initial assessment within **5 business days**.

## Security Scope

VaeaNTT is a **cryptographic primitive library**. The following are in scope:

### In Scope

- **Timing side-channels**: Data-dependent branches or memory access patterns in NTT operations
- **Incorrect modular arithmetic**: Bugs that produce wrong results (could break cryptographic protocols)
- **Integer overflow**: Arithmetic overflow in butterfly operations or reduction
- **Memory safety**: Undefined behavior in `unsafe` blocks (NEON intrinsics)

### Out of Scope

- **Misuse of API**: Using non-prime moduli, wrong polynomial sizes, etc. (covered by `try_new()` validation)
- **Physical side-channels**: Power analysis, EM emissions (hardware-level, not software-mitigable)
- **Denial of service**: Large polynomial sizes causing slow operations (expected behavior)

## Constant-Time Guarantees

VaeaNTT's NTT32 pipeline is designed to be **constant-time by construction**:

- All modular arithmetic uses branchless operations
- No data-dependent branches in butterfly stages
- No data-dependent memory access patterns
- Harvey lazy reduction avoids conditional subtraction

These properties are validated with [dudect](https://github.com/oreparaz/dudect) statistical testing.

## Disclosure Policy

- We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure)
- Fixes will be released as patch versions (e.g., 0.1.1)
- Security advisories will be published via GitHub Security Advisories
- Credit will be given to reporters (unless they prefer anonymity)
