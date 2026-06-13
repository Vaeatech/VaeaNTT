// Copyright (C) 2024-2026 Vaea SAS
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of VaeaNTT.
//
// VaeaNTT is free software: you can redistribute it and/or modify it under
// the terms of the GNU Affero General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// VaeaNTT is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public
// License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with VaeaNTT. If not, see <https://www.gnu.org/licenses/>.

fn main() {
    // External C/ASM sources for competitive benchmarks only.
    // These directories are .gitignored and not shipped on crates.io.
    // The build succeeds without them — only vs_pqclean / vs_libcrux
    // benchmarks require them.

    #[cfg(target_arch = "aarch64")]
    {
        if std::path::Path::new("pqclean_ntt/ntt.c").exists() {
            cc::Build::new()
                .file("pqclean_ntt/ntt.c")
                .file("pqclean_ntt/__asm_NTT.S")
                .file("pqclean_ntt/__asm_iNTT.S")
                .include("pqclean_ntt")
                .flag("-O3")
                .compile("pqclean_ntt");
        }

        if std::path::Path::new("mlkem_ntt/wrapper.c").exists() {
            cc::Build::new()
                .file("mlkem_ntt/wrapper.c")
                .file("mlkem_ntt/ntt_aarch64_asm.S")
                .file("mlkem_ntt/intt_aarch64_asm.S")
                .include("mlkem_ntt")
                .flag("-O3")
                .compile("mlkem_ntt");
        }
    }
}
