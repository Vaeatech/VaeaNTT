

export { VaeaNtt32 } from "./VaeaNtt32.mjs"

export { VaeaPrimeList } from "./VaeaPrimeList.mjs"

export { VaeaNttError } from "./VaeaNttError.mjs"

import wasm from "./diplomat-wasm.mjs";
import {FUNCTION_PARAM_ALLOC, internalConstructor} from "./diplomat-runtime.mjs";

FUNCTION_PARAM_ALLOC.reserve(internalConstructor, wasm, 28);
