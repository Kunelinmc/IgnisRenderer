import type {
	AnimationAnyStateTransition,
	AnimationParameterDefinition,
	AnimationStateDefinition,
	AnimationSubStateMachineDefinition,
	AnimationTransition,
	AnimationTransitionCondition,
} from "./types";

export interface AnimationStateMachineOptions {
	name?: string;
	parameters?: AnimationParameterDefinition[];
	states?: AnimationStateDefinition[];
	transitions?: AnimationTransition[];
	anyStateTransitions?: AnimationAnyStateTransition[];
	subStateMachines?: AnimationSubStateMachineDefinition[];
	initialState?: string;
}

export interface AnimationTransitionState {
	id: number;
	from: string;
	to: string;
	duration: number;
	elapsed: number;
	priority: number;
	interruptible: boolean;
}

interface CompiledTransition {
	from: string;
	to: string;
	duration: number;
	hasExitTime: boolean;
	exitTime: number;
	conditions: AnimationTransitionCondition[];
	priority: number;
	canInterrupt: boolean;
	interruptible: boolean;
	order: number;
}

interface TransitionMatch {
	transition: CompiledTransition;
	from: string;
	to: string;
	sourceIndex: number;
}

const ANY_STATE_REFERENCE = "__any__";

export class AnimationStateMachine {
	public readonly name: string;
	private _parameters = new Map<string, number | boolean>();
	private _parameterTypes = new Map<
		string,
		AnimationParameterDefinition["type"]
	>();
	private _states = new Map<string, AnimationStateDefinition>();
	private _transitions: CompiledTransition[] = [];
	private _anyStateTransitions: CompiledTransition[] = [];
	private _statePrefixByAlias = new Map<string, string>();
	private _stateEntryByAlias = new Map<string, string>();
	private _currentState: string | null = null;
	private _transitionState: AnimationTransitionState | null = null;
	private _transitionOrder = 0;
	private _transitionSerial = 0;

	constructor(options: AnimationStateMachineOptions = {}) {
		this.name = options.name ?? "AnimationStateMachine";
		for (const parameter of options.parameters ?? []) {
			this._parameterTypes.set(parameter.name, parameter.type);
			if (parameter.defaultValue !== undefined) {
				this._parameters.set(parameter.name, parameter.defaultValue);
			}
		}

		this._compileStateDefinitions(options.states ?? [], "");
		this._compileSubStateMachines(options.subStateMachines ?? [], "");
		this._compileTransitions(options.transitions ?? [], "", false);
		this._compileTransitions(options.anyStateTransitions ?? [], "", true);

		const initialReference =
			options.initialState ??
			options.states?.[0]?.name ??
			options.subStateMachines?.[0]?.name;
		if (initialReference) {
			const resolved = this._resolveStateReference(initialReference);
			if (!resolved) {
				throw new Error(
					`AnimationStateMachine "${this.name}" has invalid initial state "${initialReference}"`
				);
			}
			this._currentState = resolved;
		}
	}

	public get currentState(): AnimationStateDefinition | null {
		if (!this._currentState) return null;
		return this._states.get(this._currentState) ?? null;
	}

	public get currentStateName(): string | null {
		return this._currentState;
	}

	public get transitionState(): AnimationTransitionState | null {
		return this._transitionState;
	}

	public setParameter(name: string, value: number | boolean): void {
		this._parameters.set(name, value);
	}

	public getParameter(name: string): number | boolean | undefined {
		return this._parameters.get(name);
	}

	public addState(state: AnimationStateDefinition): void {
		if (this._statePrefixByAlias.has(state.name)) {
			throw new Error(
				`AnimationStateMachine "${this.name}" state "${state.name}" collides with sub state machine alias`
			);
		}
		this._states.set(state.name, state);
		if (!this._currentState) {
			this._currentState = state.name;
		}
	}

	public getStateDefinition(name: string): AnimationStateDefinition | null {
		const resolved = this._resolveStateReference(name);
		if (!resolved) return null;
		return this._states.get(resolved) ?? null;
	}

	public addTransition(transition: AnimationTransition): void {
		this._compileTransitions([transition], "", false);
	}

	public addAnyStateTransition(transition: AnimationAnyStateTransition): void {
		this._compileTransitions([transition], "", true);
	}

	public update(normalizedTime: number, deltaSeconds: number): void {
		if (!this._currentState) return;
		const dt = Math.max(0, deltaSeconds);

		if (this._transitionState) {
			this._transitionState.elapsed += dt;
			if (this._transitionState.elapsed >= this._transitionState.duration) {
				this._currentState = this._transitionState.to;
				this._transitionState = null;
			}
		}

		if (this._transitionState) {
			if (!this._transitionState.interruptible) return;
			const interrupt = this._findTransitionMatch(
				[this._transitionState.to, this._transitionState.from],
				normalizedTime,
				true
			);
			if (!interrupt) return;
			this._currentState = interrupt.from;
			this._startTransition(interrupt);
			return;
		}

		const next = this._findTransitionMatch(
			[this._currentState],
			normalizedTime,
			false
		);
		if (next) {
			this._startTransition(next);
		}
	}

	private _evaluateConditions(
		conditions: AnimationTransitionCondition[]
	): boolean {
		if (conditions.length === 0) return true;
		for (const condition of conditions) {
			const currentValue = this._parameters.get(condition.parameter);
			switch (condition.operator) {
				case "trigger":
					if (currentValue !== true) return false;
					break;
				case "==":
					if (currentValue !== condition.value) return false;
					break;
				case "!=":
					if (currentValue === condition.value) return false;
					break;
				case ">":
					if (!(Number(currentValue) > Number(condition.value))) return false;
					break;
				case ">=":
					if (!(Number(currentValue) >= Number(condition.value))) return false;
					break;
				case "<":
					if (!(Number(currentValue) < Number(condition.value))) return false;
					break;
				case "<=":
					if (!(Number(currentValue) <= Number(condition.value))) return false;
					break;
			}
		}
		return true;
	}

	private _consumeTriggersForConditions(
		conditions: AnimationTransitionCondition[]
	): void {
		for (const condition of conditions) {
			if (condition.operator !== "trigger") continue;
			if (this._parameterTypes.get(condition.parameter) !== "trigger") continue;
			if (this._parameters.get(condition.parameter) === true) {
				this._parameters.set(condition.parameter, false);
			}
		}
	}

	public get parameterValues(): ReadonlyMap<string, number | boolean> {
		return this._parameters;
	}

	private _compileStateDefinitions(
		states: AnimationStateDefinition[],
		scopePrefix: string
	): void {
		for (const state of states) {
			const fullName = this._resolveScopedReference(scopePrefix, state.name);
			if (this._statePrefixByAlias.has(fullName)) {
				throw new Error(
					`AnimationStateMachine "${this.name}" state "${fullName}" collides with sub state machine alias`
				);
			}
			if (this._states.has(fullName)) {
				throw new Error(
					`AnimationStateMachine "${this.name}" has duplicate state "${fullName}"`
				);
			}
			this._states.set(fullName, {
				...state,
				name: fullName,
			});
		}
	}

	private _compileSubStateMachines(
		subStateMachines: AnimationSubStateMachineDefinition[],
		scopePrefix: string
	): void {
		for (const subStateMachine of subStateMachines) {
			const alias = this._resolveScopedReference(scopePrefix, subStateMachine.name);
			if (this._states.has(alias)) {
				throw new Error(
					`AnimationStateMachine "${this.name}" sub state machine "${alias}" collides with a concrete state`
				);
			}
			if (this._statePrefixByAlias.has(alias)) {
				throw new Error(
					`AnimationStateMachine "${this.name}" has duplicate sub state machine "${alias}"`
				);
			}

			const nestedPrefix = `${alias}/`;
			this._statePrefixByAlias.set(alias, nestedPrefix);
			this._compileStateDefinitions(subStateMachine.states, nestedPrefix);
			this._compileSubStateMachines(
				subStateMachine.subStateMachines ?? [],
				nestedPrefix
			);

			const entryReference =
				subStateMachine.initialState ?? subStateMachine.states?.[0]?.name;
			if (!entryReference) {
				throw new Error(
					`AnimationStateMachine "${this.name}" sub state machine "${alias}" requires at least one state or an explicit initialState`
				);
			}
			const scopedEntry = this._resolveScopedReference(
				nestedPrefix,
				entryReference
			);
			const resolvedEntry = this._resolveStateReference(scopedEntry);
			if (!resolvedEntry) {
				throw new Error(
					`AnimationStateMachine "${this.name}" sub state machine "${alias}" has invalid initial state "${entryReference}"`
				);
			}
			this._stateEntryByAlias.set(alias, resolvedEntry);

			this._compileTransitions(subStateMachine.transitions ?? [], nestedPrefix, false);
			this._compileTransitions(
				subStateMachine.anyStateTransitions ?? [],
				nestedPrefix,
				true
			);
		}
	}

	private _compileTransitions(
		transitions: Array<AnimationTransition | AnimationAnyStateTransition>,
		scopePrefix: string,
		forceAnyState: boolean
	): void {
		for (const transition of transitions) {
			const transitionFrom = forceAnyState
				? ANY_STATE_REFERENCE
				: (transition as AnimationTransition).from;
			const fromReference =
				transitionFrom === "*"
					? ANY_STATE_REFERENCE
					: this._resolveScopedReference(scopePrefix, transitionFrom);
			const toReference = this._resolveScopedReference(scopePrefix, transition.to);
			const compiled: CompiledTransition = {
				from: fromReference,
				to: toReference,
				duration: Math.max(0, transition.duration),
				hasExitTime: transition.hasExitTime === true,
				exitTime: transition.exitTime ?? 1,
				conditions: transition.conditions ?? [],
				priority: transition.priority ?? 0,
				canInterrupt: transition.canInterrupt === true,
				interruptible: transition.interruptible !== false,
				order: this._transitionOrder++,
			};
			this._validateTransition(compiled);
			if (compiled.from === ANY_STATE_REFERENCE) {
				this._anyStateTransitions.push(compiled);
			} else {
				this._transitions.push(compiled);
			}
		}
	}

	private _validateTransition(transition: CompiledTransition): void {
		if (
			transition.from !== ANY_STATE_REFERENCE &&
			!this._isKnownStateReference(transition.from)
		) {
			throw new Error(
				`AnimationStateMachine "${this.name}" transition source "${transition.from}" does not exist`
			);
		}
		const resolvedTarget = this._resolveStateReference(transition.to);
		if (!resolvedTarget) {
			throw new Error(
				`AnimationStateMachine "${this.name}" transition target "${transition.to}" does not exist`
			);
		}
	}

	private _findTransitionMatch(
		sourceStates: string[],
		normalizedTime: number,
		interruptOnly: boolean
	): TransitionMatch | null {
		const candidates: TransitionMatch[] = [];
		for (let sourceIndex = 0; sourceIndex < sourceStates.length; sourceIndex++) {
			const sourceState = sourceStates[sourceIndex];
			for (const transition of this._transitions) {
				const match = this._evaluateTransitionCandidate(
					transition,
					sourceState,
					normalizedTime,
					interruptOnly,
					sourceIndex
				);
				if (match) candidates.push(match);
			}
			for (const transition of this._anyStateTransitions) {
				const match = this._evaluateTransitionCandidate(
					transition,
					sourceState,
					normalizedTime,
					interruptOnly,
					sourceIndex
				);
				if (match) candidates.push(match);
			}
		}

		if (candidates.length === 0) return null;
		candidates.sort((left, right) => {
			const priorityDelta = right.transition.priority - left.transition.priority;
			if (priorityDelta !== 0) return priorityDelta;
			const orderDelta = left.transition.order - right.transition.order;
			if (orderDelta !== 0) return orderDelta;
			return left.sourceIndex - right.sourceIndex;
		});
		return candidates[0];
	}

	private _evaluateTransitionCandidate(
		transition: CompiledTransition,
		sourceState: string,
		normalizedTime: number,
		interruptOnly: boolean,
		sourceIndex: number
	): TransitionMatch | null {
		if (!this._matchesTransitionSource(transition.from, sourceState)) {
			return null;
		}
		if (interruptOnly && !transition.canInterrupt) {
			return null;
		}
		if (transition.hasExitTime && normalizedTime < transition.exitTime) {
			return null;
		}
		if (!this._evaluateConditions(transition.conditions)) {
			return null;
		}
		const resolvedTarget = this._resolveStateReference(transition.to);
		if (!resolvedTarget) {
			throw new Error(
				`AnimationStateMachine "${this.name}" transition target "${transition.to}" does not exist`
			);
		}
		return {
			transition,
			from: sourceState,
			to: resolvedTarget,
			sourceIndex,
		};
	}

	private _startTransition(match: TransitionMatch): void {
		this._transitionState = {
			id: ++this._transitionSerial,
			from: match.from,
			to: match.to,
			duration: match.transition.duration,
			elapsed: 0,
			priority: match.transition.priority,
			interruptible: match.transition.interruptible,
		};
		this._consumeTriggersForConditions(match.transition.conditions);
	}

	private _matchesTransitionSource(
		reference: string,
		sourceState: string
	): boolean {
		if (reference === ANY_STATE_REFERENCE) return true;
		if (reference === sourceState) return true;
		const prefix = this._statePrefixByAlias.get(reference);
		if (!prefix) return false;
		return sourceState.startsWith(prefix);
	}

	private _resolveScopedReference(scopePrefix: string, reference: string): string {
		if (reference === "*" || reference === ANY_STATE_REFERENCE) {
			return ANY_STATE_REFERENCE;
		}
		if (reference.includes("/")) return reference;
		return `${scopePrefix}${reference}`;
	}

	private _isKnownStateReference(reference: string): boolean {
		return (
			this._states.has(reference) || this._statePrefixByAlias.has(reference)
		);
	}

	private _resolveStateReference(reference: string): string | null {
		let resolved = reference;
		const visited = new Set<string>();
		while (this._stateEntryByAlias.has(resolved)) {
			if (visited.has(resolved)) {
				throw new Error(
					`AnimationStateMachine "${this.name}" has cyclic sub state machine entry for "${reference}"`
				);
			}
			visited.add(resolved);
			resolved = this._stateEntryByAlias.get(resolved)!;
		}
		return this._states.get(resolved) ? resolved : null;
	}
}
