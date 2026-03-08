import type {
	AnimationParameterDefinition,
	AnimationStateDefinition,
	AnimationTransition,
	AnimationTransitionCondition,
} from "./types";

export interface AnimationStateMachineOptions {
	name?: string;
	parameters?: AnimationParameterDefinition[];
	states?: AnimationStateDefinition[];
	transitions?: AnimationTransition[];
	initialState?: string;
}

export interface AnimationTransitionState {
	from: string;
	to: string;
	duration: number;
	elapsed: number;
}

export class AnimationStateMachine {
	public readonly name: string;
	private _parameters = new Map<string, number | boolean>();
	private _parameterTypes = new Map<
		string,
		AnimationParameterDefinition["type"]
	>();
	private _states = new Map<string, AnimationStateDefinition>();
	private _transitions: AnimationTransition[] = [];
	private _currentState: string | null = null;
	private _transitionState: AnimationTransitionState | null = null;

	constructor(options: AnimationStateMachineOptions = {}) {
		this.name = options.name ?? "AnimationStateMachine";
		for (const parameter of options.parameters ?? []) {
			this._parameterTypes.set(parameter.name, parameter.type);
			if (parameter.defaultValue !== undefined) {
				this._parameters.set(parameter.name, parameter.defaultValue);
			}
		}
		for (const state of options.states ?? []) {
			this._states.set(state.name, state);
		}
		this._transitions = options.transitions ? [...options.transitions] : [];
		this._currentState =
			options.initialState ?? options.states?.[0]?.name ?? this._currentState;
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
		this._states.set(state.name, state);
		if (!this._currentState) {
			this._currentState = state.name;
		}
	}

	public getStateDefinition(name: string): AnimationStateDefinition | null {
		return this._states.get(name) ?? null;
	}

	public addTransition(transition: AnimationTransition): void {
		this._transitions.push(transition);
	}

	public update(normalizedTime: number, deltaSeconds: number): void {
		if (!this._currentState) return;

		if (this._transitionState) {
			this._transitionState.elapsed += Math.max(0, deltaSeconds);
			if (this._transitionState.elapsed >= this._transitionState.duration) {
				this._currentState = this._transitionState.to;
				this._transitionState = null;
			}
			this._consumeTriggers();
			return;
		}

		for (const transition of this._transitions) {
			if (transition.from !== this._currentState) continue;
			if (
				transition.hasExitTime &&
				normalizedTime < (transition.exitTime ?? 1)
			) {
				continue;
			}
			if (!this._evaluateConditions(transition.conditions ?? [])) {
				continue;
			}
			this._transitionState = {
				from: transition.from,
				to: transition.to,
				duration: Math.max(0, transition.duration),
				elapsed: 0,
			};
			this._consumeTriggers();
			return;
		}

		this._consumeTriggers();
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

	private _consumeTriggers(): void {
		for (const [name, type] of this._parameterTypes.entries()) {
			if (type === "trigger" && this._parameters.get(name) === true) {
				this._parameters.set(name, false);
			}
		}
	}

	public get parameterValues(): ReadonlyMap<string, number | boolean> {
		return this._parameters;
	}
}
