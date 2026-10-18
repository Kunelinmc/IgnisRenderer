function binary(operator, left, right) {
	if (operator === "*" && Array.isArray(left) && Array.isArray(left[0])) {
		return right.map((_, row) => left.reduce((sum, column, index) =>
			Math.fround(sum + Math.fround(column[row] * right[index])), 0));
	}
	if (Array.isArray(left) || Array.isArray(right)) {
		const lanes = Array.isArray(left) ? left : right;
		return lanes.map((_, index) => binary(operator,
			Array.isArray(left) ? left[index] : left,
			Array.isArray(right) ? right[index] : right));
	}
	switch (operator) {
		case "+": return Math.fround(left + right);
		case "-": return Math.fround(left - right);
		case "*": return Math.fround(left * right);
		case "/": return Math.fround(left / right);
		case "<": return left < right;
		case ">": return left > right;
		case "<=": return left <= right;
		case ">=": return left >= right;
		case "&&": return left && right;
		case "||": return left || right;
		default: throw new Error(`Unsupported WGSL math operator: ${operator}`);
	}
}

function compileExpression(source) {
	const tokens = [];
	const pattern = /\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?|[A-Za-z_]\w*|&&|\|\||<=|>=|[()[\].,+*/<>-])/y;
	let offset = 0;
	while (offset < source.trimEnd().length) {
		pattern.lastIndex = offset;
		const match = pattern.exec(source);
		if (!match) throw new Error(`Unsupported WGSL math syntax: ${source.slice(offset)}`);
		tokens.push(match[1]);
		offset = pattern.lastIndex;
	}
	let index = 0;
	const expect = (token) => {
		if (tokens[index++] !== token) throw new Error(`Expected '${token}' in ${source}`);
	};
	function parse(minimumPrecedence = -3) {
		const token = tokens[index++];
		let expression;
		if (token === "-") {
			const operand = parse(3);
			expression = (scope) => binary("*", -1, operand(scope));
		} else if (token === "(") {
			expression = parse();
			expect(")");
		} else if (/^\d/.test(token)) {
			expression = () => Math.fround(Number(token));
		} else if (/^[A-Za-z_]\w*$/.test(token)) {
			expression = (scope) => {
				if (!(token in scope)) throw new Error(`Unknown WGSL math name: ${token}`);
				return scope[token];
			};
		} else {
			throw new Error(`Unexpected WGSL math token: ${token}`);
		}
		while (index < tokens.length) {
			const next = tokens[index];
			const left = expression;
			if (next === ".") {
				index++;
				const lanes = tokens[index++];
				if (!/^[xyzw]{1,4}$/.test(lanes)) throw new Error(`Unsupported swizzle: ${lanes}`);
				expression = (scope) => {
					const vector = left(scope);
					const values = [...lanes].map((lane) => vector["xyzw".indexOf(lane)]);
					return values.length === 1 ? values[0] : values;
				};
			} else if (next === "[") {
				index++;
				const subscript = parse();
				expect("]");
				expression = (scope) => left(scope)[subscript(scope)];
			} else if (next === "(") {
				index++;
				const args = [];
				if (tokens[index] !== ")") {
					do {
						args.push(parse());
						if (tokens[index] !== ",") break;
						index++;
					} while (true);
				}
				expect(")");
				expression = (scope) => left(scope)(...args.map((argument) => argument(scope)));
			} else {
				const precedence = {
					"||": -3, "&&": -2, "<": -1, ">": -1, "<=": -1, ">=": -1,
					"+": 1, "-": 1, "*": 2, "/": 2,
				}[next];
				if (precedence === undefined || precedence < minimumPrecedence) break;
				index++;
				const right = parse(precedence + 1);
				expression = (scope) => binary(next, left(scope), right(scope));
			}
		}
		return expression;
	}
	const expression = parse();
	if (index !== tokens.length) throw new Error(`Unexpected token in WGSL math: ${tokens[index]}`);
	return expression;
}

// Only straight-line scalar/vector/matrix math is supported, not general WGSL or GPU validation.
// Interpret the shader asset itself so regressions cannot hide behind copied test formulas.
export function compileWGSLMath(source, functionNames) {
	const vector = (length) => (...values) => {
		const flattened = values.flat();
		if (flattened.length === 1) return Array(length).fill(Math.fround(flattened[0]));
		if (flattened.length !== length) throw new Error(`Expected ${length} WGSL vector lanes.`);
		return flattened.map(Math.fround);
	};
	const dot = (left, right) => left.reduce((sum, value, index) =>
		Math.fround(sum + Math.fround(value * right[index])), 0);
	const functions = {
		dot,
		abs: (value) => Array.isArray(value)
			? value.map((lane) => Math.fround(Math.abs(lane)))
			: Math.fround(Math.abs(value)),
		max: (left, right) => Math.fround(Math.max(left, right)),
		min: (left, right) => Math.fround(Math.min(left, right)),
		clamp: (value, low, high) => Math.fround(Math.min(high, Math.max(low, value))),
		select: (left, right, condition) => condition ? right : left,
		mix: (left, right, weight) => binary("+",
			binary("*", left, binary("-", 1, weight)), binary("*", right, weight)),
		normalize: (value) => binary("/", value, Math.fround(Math.sqrt(dot(value, value)))),
		vec2: vector(2), vec3: vector(3), vec4: vector(4),
	};
	const code = source.replace(/\/\/[^\r\n]*/g, "").replace(/<f32>/g, "");
	for (const match of code.matchAll(/fn\s+(\w+)\s*\(([^)]*)\)\s*->[^{}]+\{([^{}]*)\}/g)) {
		const [, name, signature, body] = match;
		const parameters = [...signature.matchAll(/(\w+)\s*:/g)].map((entry) => entry[1]);
		const statements = body.split(";").map((line) => line.trim()).filter(Boolean).map((line) => {
			const assignment = line.match(/^let\s+(\w+)\s*=\s*([\s\S]+)$/);
			if (assignment) return { name: assignment[1], evaluate: compileExpression(assignment[2]) };
			if (line.startsWith("return ")) return { evaluate: compileExpression(line.slice(7)) };
			throw new Error(`Unsupported WGSL math statement: ${line}`);
		});
		functions[name] = (...args) => {
			const scope = Object.create(functions);
			parameters.forEach((parameter, index) => { scope[parameter] = args[index]; });
			for (const statement of statements) {
				const result = statement.evaluate(scope);
				if (!statement.name) return result;
				scope[statement.name] = result;
			}
		};
	}
	return Object.fromEntries(functionNames.map((name) => {
		if (!functions[name]) throw new Error(`Missing WGSL math function: ${name}`);
		return [name, functions[name]];
	}));
}
