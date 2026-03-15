export class IdGenerator {
	private static _counts: Map<string, number> = new Map();

	public static nextId(prefix: string = "id"): string {
		const count = (this._counts.get(prefix) ?? 0) + 1;
		this._counts.set(prefix, count);
		return `${prefix}_${count}`;
	}
}
