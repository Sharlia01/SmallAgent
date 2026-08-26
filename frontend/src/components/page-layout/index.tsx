import classNames from 'classnames'
import { PropsWithChildren, ReactNode, useState } from 'react'
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

  const setRightCollapsed = (collapsed: boolean) => {
    setInternalRightCollapsed(collapsed)
    onRightCollapsedChange?.(collapsed)
  }

  return (
    <div
      className={classNames('com-page-layout', className, {
        'com-page-layout--right-collapsed': right && rightCollapsed,
      })}
      {...rest}
    >
      <div className="com-page-layout__main">
        <div className="com-page-layout__main-content">{children}</div>

        <div className="com-page-layout__sender">{sender}</div>
      </div>
      {right ? (
        <>
          <button
            type="button"
            className="com-page-layout__right-toggle"
            aria-label={rightCollapsed ? '展开右侧栏' : '折叠右侧栏'}
            aria-expanded={!rightCollapsed}
            title={rightCollapsed ? '展开右侧栏' : '折叠右侧栏'}
            onClick={() => setRightCollapsed(!rightCollapsed)}
          >
            {rightCollapsed ? '‹' : '›'}
          </button>
          <div className="com-page-layout__right" aria-hidden={rightCollapsed}>
            <div className="com-page-layout__right-inner">{right}</div>
          </div>
        </>
      ) : null}
    </div>
  )
}
