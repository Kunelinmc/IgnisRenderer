export class ComponentStore<T> {
	private _denseEntities: number[] = []
	private _denseData: T[] = []
	private _sparse = new Map<number, number>()

	public get size(): number {
		return this._denseEntities.length
	}

	public has(entity: number): boolean {
		return this._sparse.has(entity)
	}

	public get(entity: number): T | undefined {
		const index = this._sparse.get(entity)
		if (index === undefined) return undefined
		return this._denseData[index]
	}

	public set(entity: number, data: T): void {
		const existing = this._sparse.get(entity)
		if (existing !== undefined) {
			this._denseData[existing] = data
			return
		}

		const index = this._denseEntities.length
		this._sparse.set(entity, index)
		this._denseEntities.push(entity)
		this._denseData.push(data)
	}

	public delete(entity: number): boolean {
		const index = this._sparse.get(entity)
		if (index === undefined) return false

		const lastIndex = this._denseEntities.length - 1
		const lastEntity = this._denseEntities[lastIndex]
		const lastData = this._denseData[lastIndex]

		this._denseEntities[index] = lastEntity
		this._denseData[index] = lastData
		this._sparse.set(lastEntity, index)

		this._denseEntities.pop()
		this._denseData.pop()
		this._sparse.delete(entity)
		return true
	}

	public clear(): void {
		this._denseEntities.length = 0
		this._denseData.length = 0
		this._sparse.clear()
	}

	public entities(): readonly number[] {
		return this._denseEntities
	}

	public values(): readonly T[] {
		return this._denseData
	}

	public entries(): Array<[number, T]> {
		const result: Array<[number, T]> = []
		for (let i = 0; i < this._denseEntities.length; i++) {
			result.push([this._denseEntities[i], this._denseData[i]])
		}
		return result
	}
}
