import { AxiosRequestConfig, AxiosResponse, CanceledError } from 'axios'
import { ResponseError } from '../error'
import { IRequestPlugin } from './plugin'

const NETWORK_ERROR_MAP = {
  // '400': '请求参数错误',
  // '401': '尚未授权，请重新登录',
  // '403': '拒绝访问',
  // '404': '请求的资源不存在',
  // '405': '不支持此请求方法',
  // '408': '请求超时',
  429: '请求过于频繁，请稍后再试',
  // '500': '服务器内部错误',
  // '501': '服务尚未实现此功能',
  // '502': '网关响应异常',
  // '503': '服务暂时不可用',
  // '504': '网关请求超时',
  // '505': '不支持此 HTTP 协议版本',
}

export const errorToastPlugin: IRequestPlugin = {
  postinstall(instance) {
    instance.interceptors.response.use(
      (response) => response,
      (error) => {
        const response = error.response as AxiosResponse<any> | undefined
        const config = (response?.config ?? error?.config) as AxiosRequestConfig

        if (config && !config.errorToast) return Promise.reject(error)

        // CanceledError 主要来源于 repeat.ts 取消重复请求
        // 该错误不应展示给用户
        if (error instanceof CanceledError) return Promise.reject(error)

        const status = response?.status ?? ''
        const message =
          error instanceof ResponseError
            ? error.message
            : NETWORK_ERROR_MAP[status as keyof typeof NETWORK_ERROR_MAP] ||
              response?.data?.message ||
              response?.data?.detail ||
              response?.data?.error ||
              error.message ||
              '请求错误'

        window.$app.message.error(
          typeof message === 'string' ? message : JSON.stringify(message),
        )

        return Promise.reject(error)
      },
    )
  },
}
