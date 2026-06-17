import { ConditionParseError } from "../../foundation/Error";
import {
	isIdentifierPartCharacter,
	isIdentifierStartCharacter,
	isWhitespaceCharacter,
} from "./runtimeShared";

type ConditionTokenKind =
	| "eof"
	| "identifier"
	| "number"
	| "operator"
	| "leftParen"
	| "rightParen";

interface ConditionToken {
	kind: ConditionTokenKind;
	text: string;
	column: number;
}

interface DirectiveConditionParserOptions {
	expression: string;
	baseColumn: number;
	isDefined: (identifier: string) => boolean;
	resolveIdentifier: (identifier: string) => bigint;
}

export class DirectiveConditionParser {
	private _expression: string;
	private _baseColumn: number;
	private _isDefined: (identifier: string) => boolean;
	private _resolveIdentifier: (identifier: string) => bigint;
	private _tokens: ConditionToken[];
	private _index = 0;

	public constructor(options: DirectiveConditionParserOptions) {
		this._expression = options.expression;
		this._baseColumn = options.baseColumn;
		this._isDefined = options.isDefined;
		this._resolveIdentifier = options.resolveIdentifier;
		this._tokens = this._tokenize();
	}

	public parse(): bigint {
		const value = this._parseLogicalOr();
		const token = this._peek();
		if (token.kind !== "eof") {
			throw new ConditionParseError(
				`Unexpected token "${token.text}" in directive expression.`,
				token.column
			);
		}
		return value;
	}

	private _parseLogicalOr(): bigint {
		let value = this._parseLogicalAnd();
		while (this._matchOperator("||")) {
			const right = this._parseLogicalAnd();
			value = value !== 0n || right !== 0n ? 1n : 0n;
		}
		return value;
	}

	private _parseLogicalAnd(): bigint {
		let value = this._parseEquality();
		while (this._matchOperator("&&")) {
			const right = this._parseEquality();
			value = value !== 0n && right !== 0n ? 1n : 0n;
		}
		return value;
	}

	private _parseEquality(): bigint {
		let value = this._parseRelational();
		while (true) {
			if (this._matchOperator("==")) {
				const right = this._parseRelational();
				value = value === right ? 1n : 0n;
				continue;
			}
			if (this._matchOperator("!=")) {
				const right = this._parseRelational();
				value = value !== right ? 1n : 0n;
				continue;
			}
			return value;
		}
	}

	private _parseRelational(): bigint {
		let value = this._parseAdditive();
		while (true) {
			if (this._matchOperator("<")) {
				const right = this._parseAdditive();
				value = value < right ? 1n : 0n;
				continue;
			}
			if (this._matchOperator(">")) {
				const right = this._parseAdditive();
				value = value > right ? 1n : 0n;
				continue;
			}
			if (this._matchOperator("<=")) {
				const right = this._parseAdditive();
				value = value <= right ? 1n : 0n;
				continue;
			}
			if (this._matchOperator(">=")) {
				const right = this._parseAdditive();
				value = value >= right ? 1n : 0n;
				continue;
			}
			return value;
		}
	}

	private _parseAdditive(): bigint {
		let value = this._parseMultiplicative();
		while (true) {
			if (this._matchOperator("+")) {
				value += this._parseMultiplicative();
				continue;
			}
			if (this._matchOperator("-")) {
				value -= this._parseMultiplicative();
				continue;
			}
			return value;
		}
	}

	private _parseMultiplicative(): bigint {
		let value = this._parseUnary();
		while (true) {
			if (this._matchOperator("*")) {
				value *= this._parseUnary();
				continue;
			}
			if (this._matchOperator("/")) {
				const right = this._parseUnary();
				if (right === 0n) {
					throw new ConditionParseError(
						"Division by zero in directive condition expression.",
						this._previous().column
					);
				}
				value /= right;
				continue;
			}
			if (this._matchOperator("%")) {
				const right = this._parseUnary();
				if (right === 0n) {
					throw new ConditionParseError(
						"Modulo by zero in directive condition expression.",
						this._previous().column
					);
				}
				value %= right;
				continue;
			}
			return value;
		}
	}

	private _parseUnary(): bigint {
		if (this._matchOperator("!")) {
			const value = this._parseUnary();
			return value === 0n ? 1n : 0n;
		}
		if (this._matchOperator("-")) {
			return -this._parseUnary();
		}
		if (this._matchOperator("+")) {
			return this._parseUnary();
		}
		return this._parsePrimary();
	}

	private _parsePrimary(): bigint {
		const token = this._peek();
		if (token.kind === "number") {
			this._consume();
			return this._parseIntegerLiteral(token);
		}
		if (token.kind === "identifier") {
			this._consume();
			if (token.text === "defined") {
				return this._parseDefinedOperator(token.column);
			}
			return this._resolveIdentifier(token.text);
		}
		if (token.kind === "leftParen") {
			this._consume();
			const value = this._parseLogicalOr();
			const closing = this._peek();
			if (closing.kind !== "rightParen") {
				throw new ConditionParseError(
					`Expected ")" but got "${closing.text}".`,
					closing.column
				);
			}
			this._consume();
			return value;
		}
		throw new ConditionParseError(
			`Unexpected token "${token.text}" in directive expression.`,
			token.column
		);
	}

	private _parseDefinedOperator(operatorColumn: number): bigint {
		if (this._peek().kind === "leftParen") {
			this._consume();
			const identifier = this._peek();
			if (identifier.kind !== "identifier") {
				throw new ConditionParseError(
					`Expected identifier after "defined(" but got "${identifier.text}".`,
					identifier.column
				);
			}
			this._consume();
			const closing = this._peek();
			if (closing.kind !== "rightParen") {
				throw new ConditionParseError(
					`Expected ")" after "defined(${identifier.text}" but got "${closing.text}".`,
					closing.column
				);
			}
			this._consume();
			return this._isDefined(identifier.text) ? 1n : 0n;
		}
		const identifier = this._peek();
		if (identifier.kind !== "identifier") {
			throw new ConditionParseError(
				`Expected identifier after "defined" but got "${identifier.text}".`,
				identifier.column
			);
		}
		this._consume();
		if (identifier.column < operatorColumn) {
			throw new ConditionParseError(
				"Invalid defined() expression in directive condition.",
				operatorColumn
			);
		}
		return this._isDefined(identifier.text) ? 1n : 0n;
	}

	private _parseIntegerLiteral(token: ConditionToken): bigint {
		if (
			!/^(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|0[oO][0-7]+|[0-9]+)$/.test(token.text)
		) {
			throw new ConditionParseError(
				`Invalid integer literal "${token.text}" in directive expression.`,
				token.column
			);
		}
		try {
			return BigInt(token.text);
		} catch {
			throw new ConditionParseError(
				`Invalid integer literal "${token.text}" in directive expression.`,
				token.column
			);
		}
	}

	private _peek(): ConditionToken {
		return this._tokens[this._index];
	}

	private _previous(): ConditionToken {
		return this._tokens[Math.max(0, this._index - 1)];
	}

	private _consume(): ConditionToken {
		const current = this._tokens[this._index];
		if (this._index < this._tokens.length - 1) {
			this._index++;
		}
		return current;
	}

	private _matchOperator(operator: string): boolean {
		const token = this._peek();
		if (token.kind !== "operator" || token.text !== operator) {
			return false;
		}
		this._consume();
		return true;
	}

	private _tokenize(): ConditionToken[] {
		const tokens: ConditionToken[] = [];
		const expression = this._expression;
		let index = 0;
		while (index < expression.length) {
			const char = expression[index];
			if (isWhitespaceCharacter(char)) {
				index++;
				continue;
			}
			const column = this._baseColumn + index;
			const twoChars = expression.slice(index, index + 2);
			if (
				twoChars === "&&" ||
				twoChars === "||" ||
				twoChars === "==" ||
				twoChars === "!=" ||
				twoChars === "<=" ||
				twoChars === ">="
			) {
				tokens.push({
					kind: "operator",
					text: twoChars,
					column,
				});
				index += 2;
				continue;
			}
			if (char === "(") {
				tokens.push({
					kind: "leftParen",
					text: char,
					column,
				});
				index++;
				continue;
			}
			if (char === ")") {
				tokens.push({
					kind: "rightParen",
					text: char,
					column,
				});
				index++;
				continue;
			}
			if ("!<>+-*/%".includes(char)) {
				tokens.push({
					kind: "operator",
					text: char,
					column,
				});
				index++;
				continue;
			}
			if (isIdentifierStartCharacter(char)) {
				let end = index + 1;
				while (end < expression.length && isIdentifierPartCharacter(expression[end])) {
					end++;
				}
				tokens.push({
					kind: "identifier",
					text: expression.slice(index, end),
					column,
				});
				index = end;
				continue;
			}
			if (char >= "0" && char <= "9") {
				let end = index + 1;
				while (
					end < expression.length &&
					/[A-Za-z0-9]/.test(expression[end] ?? "")
				) {
					end++;
				}
				tokens.push({
					kind: "number",
					text: expression.slice(index, end),
					column,
				});
				index = end;
				continue;
			}
			throw new ConditionParseError(
				`Unexpected character "${char}" in directive expression.`,
				column
			);
		}
		tokens.push({
			kind: "eof",
			text: "<eof>",
			column: this._baseColumn + expression.length,
		});
		return tokens;
	}
}
