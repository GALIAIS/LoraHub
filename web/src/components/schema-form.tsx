/**
 * Renders a Pydantic JSON Schema into editable form fields.
 *
 * Scope: handles every shape produced by Pydantic v2 for the LoraHub recipe
 * (string/number/integer/boolean, enum, anyOf-with-null nullable wrappers,
 * $ref to $defs, nested objects, fixed-length tuple-like arrays). Anything
 * unsupported renders as a JSON textarea so the user can still edit it.
 */
import { Fragment } from "react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [k: string]: JsonValue }

export type SchemaNode = {
  type?: string | string[]
  title?: string
  description?: string
  default?: JsonValue
  enum?: (string | number)[]
  minimum?: number
  maximum?: number
  format?: string
  properties?: Record<string, SchemaNode>
  required?: string[]
  items?: SchemaNode | SchemaNode[]
  anyOf?: SchemaNode[]
  $ref?: string
  additionalProperties?: boolean | SchemaNode
}

export type RecipeSchema = SchemaNode & {
  $defs?: Record<string, SchemaNode>
  title?: string
}

export type FieldError = { loc: (string | number)[]; msg: string; type: string }

interface SchemaFormProps {
  schema: RecipeSchema
  value: Record<string, JsonValue>
  errors?: FieldError[]
  onChange: (next: Record<string, JsonValue>) => void
  hide?: string[]  // top-level field names to omit (e.g. backend handled in Settings)
}

export function SchemaForm({ schema, value, errors, onChange, hide }: SchemaFormProps) {
  const errorMap = buildErrorMap(errors ?? [])
  const props = schema.properties ?? {}
  const required = new Set(schema.required ?? [])

  return (
    <div className="space-y-5">
      {Object.entries(props).map(([key, node]) => {
        if (hide?.includes(key)) return null
        return (
          <FieldGroup
            key={key}
            name={key}
            schema={resolveRef(node, schema)}
            value={value[key]}
            required={required.has(key)}
            path={[key]}
            errors={errorMap}
            root={schema}
            onChange={(next) => onChange({ ...value, [key]: next })}
          />
        )
      })}
    </div>
  )
}

function FieldGroup({
  name,
  schema,
  value,
  required,
  path,
  errors,
  root,
  onChange,
}: {
  name: string
  schema: SchemaNode
  value: JsonValue | undefined
  required: boolean
  path: (string | number)[]
  errors: Map<string, string[]>
  root: RecipeSchema
  onChange: (next: JsonValue) => void
}) {
  const resolved = resolveRef(schema, root)
  const isObject = resolved.type === "object" || (resolved.properties != null)

  if (isObject) {
    return (
      <fieldset className="rounded-[6px] border border-border/60 bg-card/40 px-4 py-3 shadow-[var(--panel-shadow)]">
        <legend className="px-1.5 text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
          {humanize(resolved.title ?? name)}
          {required && <span className="ml-1 text-destructive/80">*</span>}
        </legend>
        {resolved.description && (
          <p className="text-xs text-muted-foreground mb-3 -mt-1">{resolved.description}</p>
        )}
        <ObjectFields
          schema={resolved}
          value={(value as Record<string, JsonValue> | undefined) ?? {}}
          path={path}
          errors={errors}
          root={root}
          onChange={onChange}
        />
      </fieldset>
    )
  }

  return (
    <ScalarField
      name={name}
      schema={resolved}
      value={value}
      required={required}
      path={path}
      errors={errors}
      onChange={onChange}
    />
  )
}

function ObjectFields({
  schema,
  value,
  path,
  errors,
  root,
  onChange,
}: {
  schema: SchemaNode
  value: Record<string, JsonValue>
  path: (string | number)[]
  errors: Map<string, string[]>
  root: RecipeSchema
  onChange: (next: Record<string, JsonValue>) => void
}) {
  const props = schema.properties ?? {}
  const required = new Set(schema.required ?? [])
  return (
    <div className="space-y-3.5">
      {Object.entries(props).map(([key, node]) => (
        <FieldGroup
          key={key}
          name={key}
          schema={resolveRef(node, root)}
          value={value[key]}
          required={required.has(key)}
          path={[...path, key]}
          errors={errors}
          root={root}
          onChange={(next) => onChange({ ...value, [key]: next })}
        />
      ))}
    </div>
  )
}

function ScalarField({
  name,
  schema,
  value,
  required,
  path,
  errors,
  onChange,
}: {
  name: string
  schema: SchemaNode
  value: JsonValue | undefined
  required: boolean
  path: (string | number)[]
  errors: Map<string, string[]>
  onChange: (next: JsonValue) => void
}) {
  const inner = unwrapNullable(schema)
  const fieldErrors = errors.get(path.join(".")) ?? []
  const id = `f-${path.join("-")}`
  const label = humanize(schema.title ?? name)

  // Boolean
  if (inner.type === "boolean") {
    return (
      <Row id={id} label={label} required={required} description={schema.description} errors={fieldErrors}>
        <Switch
          id={id}
          checked={Boolean(value ?? inner.default ?? false)}
          onCheckedChange={(v) => onChange(v)}
        />
      </Row>
    )
  }

  // Enum (select)
  if (inner.enum && inner.enum.length > 0) {
    return (
      <Row id={id} label={label} required={required} description={schema.description} errors={fieldErrors}>
        <select
          id={id}
          value={(value ?? inner.default ?? inner.enum[0]) as string}
          onChange={(e) => onChange(e.target.value)}
          className="h-8 rounded-[4px] border border-input bg-background px-2 text-sm font-mono"
        >
          {inner.enum.map((v) => (
            <option key={String(v)} value={String(v)}>
              {String(v)}
            </option>
          ))}
        </select>
      </Row>
    )
  }

  // Number / integer
  if (inner.type === "number" || inner.type === "integer") {
    const isInt = inner.type === "integer"
    return (
      <Row id={id} label={label} required={required} description={schema.description} errors={fieldErrors}>
        <Input
          id={id}
          type="number"
          step={isInt ? 1 : "any"}
          min={inner.minimum}
          max={inner.maximum}
          value={value === null || value === undefined ? "" : String(value)}
          onChange={(e) => {
            const raw = e.target.value
            if (raw === "") {
              onChange(null as JsonValue)
              return
            }
            const n = isInt ? parseInt(raw, 10) : parseFloat(raw)
            onChange(Number.isNaN(n) ? raw : n)
          }}
          className="font-mono w-44"
        />
      </Row>
    )
  }

  // Tuple-like fixed array of numbers (e.g. resolution: [w, h])
  if (inner.type === "array" && Array.isArray(value)) {
    return (
      <Row id={id} label={label} required={required} description={schema.description} errors={fieldErrors}>
        <div className="flex gap-2 flex-wrap">
          {(value as JsonValue[]).map((v, i) => (
            <Input
              key={i}
              type="number"
              value={String(v)}
              className="font-mono w-24"
              onChange={(e) => {
                const next = [...(value as JsonValue[])]
                const n = parseFloat(e.target.value)
                next[i] = Number.isNaN(n) ? e.target.value : n
                onChange(next)
              }}
            />
          ))}
        </div>
      </Row>
    )
  }

  // String (default), with `format: path` rendered slightly wider/monospace
  const isPath = inner.format === "path"
  return (
    <Row id={id} label={label} required={required} description={schema.description} errors={fieldErrors}>
      <Input
        id={id}
        value={(value ?? inner.default ?? "") as string}
        onChange={(e) => onChange(e.target.value === "" ? (required ? "" : null as JsonValue) : e.target.value)}
        className={cn("font-mono", isPath ? "w-full max-w-2xl" : "w-full max-w-sm")}
        placeholder={isPath ? "/path/to/file" : ""}
      />
    </Row>
  )
}

function Row({
  id,
  label,
  required,
  description,
  errors,
  children,
}: {
  id: string
  label: string
  required: boolean
  description?: string
  errors: string[]
  children: React.ReactNode
}) {
  return (
    <div className="grid grid-cols-[10rem_1fr] gap-x-4 items-start">
      <Label htmlFor={id} className="text-xs pt-2 leading-tight">
        {label}
        {required && <span className="ml-1 text-destructive/80">*</span>}
      </Label>
      <div className="min-w-0">
        {children}
        {description && (
          <p className="text-[11px] text-muted-foreground/80 mt-1">{description}</p>
        )}
        {errors.length > 0 && (
          <ul className="mt-1 text-[11px] text-destructive">
            {errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

function resolveRef(node: SchemaNode, root: RecipeSchema): SchemaNode {
  if (!node.$ref) return node
  const m = node.$ref.match(/^#\/\$defs\/(.+)$/)
  if (!m) return node
  const target = root.$defs?.[m[1]]
  return target ? { ...target, ...stripUndefined({ title: node.title, description: node.description }) } : node
}

function unwrapNullable(node: SchemaNode): SchemaNode {
  // Pydantic v2 emits `Optional[X]` as `anyOf: [{...}, {type: "null"}]`
  if (node.anyOf && node.anyOf.length === 2) {
    const [a, b] = node.anyOf
    if (b.type === "null") return { ...a, default: node.default ?? a.default, title: node.title, description: node.description }
    if (a.type === "null") return { ...b, default: node.default ?? b.default, title: node.title, description: node.description }
  }
  return node
}

function buildErrorMap(errors: FieldError[]): Map<string, string[]> {
  const m = new Map<string, string[]>()
  for (const e of errors) {
    const key = e.loc.join(".")
    const arr = m.get(key) ?? []
    arr.push(e.msg)
    m.set(key, arr)
  }
  return m
}

function stripUndefined<T extends object>(obj: T): Partial<T> {
  const out: Partial<T> = {}
  for (const [k, v] of Object.entries(obj)) {
    if (v !== undefined) (out as Record<string, unknown>)[k] = v
  }
  return out
}

function humanize(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

// Re-export Fragment so consumers don't need to import it just to satisfy
// IDE auto-complete for JSX fragments.
export { Fragment }
