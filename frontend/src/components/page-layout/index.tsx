import { useResizablePanel } from '@/hooks/use-resizable-panel'
import classNames from 'classnames'
import { CSSProperties, PropsWithChildren, ReactNode, useState } from 'react'
import './index.scss'

export default function ComPageLayout(
  props: PropsWithChildren<{
    className?: string
    right?: ReactNode
    sender?: ReactNode
    rightCollapsed?: boolean
    onRightCollapsedChange?: (collapsed: boolean) => void
  }>,
) {
  const {
    children,
    className,
    right,
    sender,
    rightCollapsed: controlledRightCollapsed,
    onRightCollapsedChange,
    ...rest
  } = props
  const [internalRightCollapsed, setInternalRightCollapsed] = useState(false)
  const rightCollapsed = controlledRightCollapsed ?? internalRightCollapsed
  const {
    size: rightWidth,
    isResizing: isRightResizing,
    handlePointerDown: handleRightPointerDown,
    handleKeyDown: handleRightKeyDown,
  } = useResizablePanel({
    defaultSize: 408,
    minSize: 280,
    maxSize: 640,
    resizeEdge: 'left',
    storageKey: 'rag-system-right-sidebar-width',
  })
  const layoutStyle = {
    '--com-page-layout-right-width': `${rightWidth}px`,
  } as CSSProperties

  const setRightCollapsed = (collapsed: boolean) => {
    setInternalRightCollapsed(collapsed)
    onRightCollapsedChange?.(collapsed)
  }

  return (
    <div
      className={classNames('com-page-layout', className, {
        'com-page-layout--right-collapsed': right && rightCollapsed,
        'com-page-layout--right-resizing': right && isRightResizing,
      })}
      style={layoutStyle}
      {...rest}
    >
      <div className="com-page-layout__main">
        <div className="com-page-layout__main-content">{children}</div>

        <div className="com-page-layout__sender">{sender}</div>
      </div>
      {right ? (
        <div className="com-page-layout__right">
          <div
            className="com-page-layout__right-resizer"
            role="separator"
            aria-label="调整右侧栏宽度"
            aria-orientation="vertical"
            aria-valuemin={280}
            aria-valuemax={640}
            aria-valuenow={rightWidth}
            tabIndex={rightCollapsed ? -1 : 0}
            title="拖动以调整右侧栏宽度"
            onPointerDown={handleRightPointerDown}
            onKeyDown={handleRightKeyDown}
          />
          <button
            type="button"
            className="com-page-layout__right-toggle"
            aria-label={rightCollapsed ? '展开右侧栏' : '折叠右侧栏'}
            aria-expanded={!rightCollapsed}
            title={rightCollapsed ? '展开右侧栏' : '折叠右侧栏'}
            onClick={() => setRightCollapsed(!rightCollapsed)}
          >
            <span
              className="com-page-layout__right-toggle-icon"
              aria-hidden="true"
            />
          </button>
          <div
            className="com-page-layout__right-viewport"
            aria-hidden={rightCollapsed}
          >
            <div className="com-page-layout__right-inner">{right}</div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
