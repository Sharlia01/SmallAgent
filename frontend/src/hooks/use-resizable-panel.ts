import {
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'

type ResizeEdge = 'left' | 'right'

interface UseResizablePanelOptions {
  defaultSize: number
  minSize: number
  maxSize: number
  resizeEdge: ResizeEdge
  storageKey: string
}

interface DragState {
  startX: number
  startSize: number
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

function readStoredSize(
  storageKey: string,
  defaultSize: number,
  minSize: number,
  maxSize: number,
) {
  try {
    const storedSize = Number(window.localStorage.getItem(storageKey))

    return Number.isFinite(storedSize) && storedSize > 0
      ? clamp(storedSize, minSize, maxSize)
      : defaultSize
  } catch {
    return defaultSize
  }
}

export function useResizablePanel({
  defaultSize,
  minSize,
  maxSize,
  resizeEdge,
  storageKey,
}: UseResizablePanelOptions) {
  const [size, setSize] = useState(() =>
    readStoredSize(storageKey, defaultSize, minSize, maxSize),
  )
  const [isResizing, setIsResizing] = useState(false)
  const dragState = useRef<DragState | null>(null)
  const direction = resizeEdge === 'right' ? 1 : -1

  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, String(size))
    } catch {
      // localStorage 不可用时仍保留当前页面内的拖拽能力。
    }
  }, [size, storageKey])

  useEffect(() => {
    if (!isResizing) return

    const previousCursor = document.body.style.cursor
    const previousUserSelect = document.body.style.userSelect

    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const handlePointerMove = (event: PointerEvent) => {
      if (!dragState.current) return

      const delta = (event.clientX - dragState.current.startX) * direction
      setSize(clamp(dragState.current.startSize + delta, minSize, maxSize))
    }

    const stopResizing = () => {
      dragState.current = null
      setIsResizing(false)
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', stopResizing)
    window.addEventListener('pointercancel', stopResizing)

    return () => {
      document.body.style.cursor = previousCursor
      document.body.style.userSelect = previousUserSelect
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', stopResizing)
      window.removeEventListener('pointercancel', stopResizing)
    }
  }, [direction, isResizing, maxSize, minSize])

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return

      event.preventDefault()
      event.currentTarget.setPointerCapture?.(event.pointerId)
      dragState.current = {
        startX: event.clientX,
        startSize: size,
      }
      setIsResizing(true)
    },
    [size],
  )

  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return

      event.preventDefault()
      const dividerMovement = event.key === 'ArrowRight' ? 12 : -12
      setSize((currentSize) =>
        clamp(currentSize + dividerMovement * direction, minSize, maxSize),
      )
    },
    [direction, maxSize, minSize],
  )

  return {
    size,
    isResizing,
    handlePointerDown,
    handleKeyDown,
  }
}
