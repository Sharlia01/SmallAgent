import 'axios'

declare module 'axios' {
  export interface AxiosRequestConfig {
    /**
     * 是否显示加载遮罩。
     * 实现位置：plugins/loading.ts。
     */
    loading?: boolean

    /**
     * 请求异常时是否显示消息提示。
     * 实现位置：plugins/error-toast.ts。
     */
    errorToast?: boolean

    /**
     * 是否取消重复请求。
     * 实现位置：plugins/repeat.ts。
     */
    cancelRepeat?: boolean
    repeatKey?: string

    /**
     * 展开接口数据，将 response.data.data 提升到 response.data。
     * 实现位置：plugins/service.ts。
     */
    unwrap?: boolean
  }

  export interface AxiosResponse<T, D> {
    /**
     * 保存展开接口数据前的原始响应。
     * 实现位置：plugins/service.ts。
     */
    _data?: {
      code: number
      msg: string
      data: T
    }
  }
}
